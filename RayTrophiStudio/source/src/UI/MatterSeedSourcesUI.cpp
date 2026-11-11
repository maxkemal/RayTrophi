#include "MatterSeedSourcesUI.h"
#include "UI/TemperatureLabel.h"
#include "DemBondControls.h"
#include "Api/MatterSeedSourcesApi.h"
#include "MaterialStateField.h"
#include "MatterSeedViewport.h"
#include "ParticleSimulation.h"
#include "imgui.h"
#include "scene_ui_substance_editor.hpp"

#include <cstdio>
#include <map>
#include <optional>

namespace RayTrophiSim::Fluid {

void drawMatterSeedSourceBounds(
    const SimulationGridDomainDesc &domain,
    const std::function<void(const Vec3 &, const Vec3 &, ImU32, float)> &draw_box) {
    if (!domain.matter_seed_sources_enabled) {
        return;
    }
    for (const auto &source : domain.matter_seed_sources) {
        if (!source.enabled || !source.mesh_object.empty()) {
            continue;
        }
        const ImU32 color = ImGui::ColorConvertFloat4ToU32(
            ImColor::HSV(static_cast<float>(source.id % 13) / 13.0f, 0.55f, 1.0f, 0.75f));
        const bool active = selectedMatterSeed(domain) == source.id;
        draw_box(source.bounds_min, source.bounds_max, active ? IM_COL32(255, 195, 75, 255) : color,
                 active ? 2.5f : 1.5f);
    }
}

void drawMatterSeedSourcesControls(const SimulationGridDomainDesc &domain) {
    if (domain.type != SimulationDomainType::Matter) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    static std::map<std::string, std::string> errors;
    auto &error = errors[domain.name];
    const auto edit = [&](const std::string &operation, uint64_t id, const nlohmann::json &patch) {
        nlohmann::json result;
        const auto status = rtapi::matterSeedSources(domain.name, operation, id, patch, result);
        error = status.ok ? std::string() : status.error;
        return status.ok;
    };

    if (ImGui::CollapsingHeader("Matter Temperature", ImGuiTreeNodeFlags_DefaultOpen)) {
        rtapi::DomainEnvironmentInfo environment;
        if (rtapi::getDomainEnvironment(domain.name, environment).ok) {
            bool override = environment.override_enabled;
            if (ImGui::Checkbox("Override domain ambient", &override)) {
                const auto status =
                    rtapi::setDomainEnvironment(domain.name, override, std::nullopt, std::nullopt);
                error = status.ok ? std::string() : status.error;
            }
            float ambient = environment.ambient_kelvin;
            ImGui::BeginDisabled(!environment.override_enabled);
            char ambient_fmt[48];
            if (ImGui::DragFloat("Domain ambient", &ambient, 0.5f, 1.0f, 3000.0f,
                    UiTemperature::kelvinFormat(ambient, ambient_fmt, sizeof(ambient_fmt), "%.2f"))) {
                const auto status =
                    rtapi::setDomainEnvironment(domain.name, std::nullopt, ambient, std::nullopt);
                error = status.ok ? std::string() : status.error;
            }
            ImGui::EndDisabled();
            char effective_text[64];
            ImGui::Text("Effective ambient: %s", UiTemperature::kelvinText(
                environment.effective_ambient_kelvin, effective_text, sizeof(effective_text), "%.2f"));
        }
        bool exchange = domain.fluid_params.thermal_liquid_enabled;
        if (ImGui::Checkbox("Enable Heat Exchange##MatterSeedThermal", &exchange)) {
            rtapi::FluidThermalPatch patch;
            patch.enabled = exchange;
            const auto status = rtapi::setFluidThermal(domain.name, patch);
            error = status.ok ? std::string() : status.error;
        }
        ImGui::TextWrapped("Environment temperature acts after birth. Each seed layer has "
                           "its own initial-temperature policy; emitters are not required.");
    }
    if (ImGui::CollapsingHeader("Initial Matter Sources", ImGuiTreeNodeFlags_DefaultOpen)) {
        nlohmann::json listing;
        const auto status =
            rtapi::matterSeedSources(domain.name, "list", 0, nlohmann::json::object(), listing);
        if (!status.ok) {
            ImGui::TextWrapped("%s", status.error.c_str());
            ImGui::PopID();
            return;
        }
        if (ImGui::Button("Add seed layer")) {
            edit("add", 0, nlohmann::json::object());
        }
        if (listing.at("enabled").get<bool>()) {
            bool automatic = domain.matter_seed_auto_capacity;
            if (ImGui::Checkbox("Automatic particle capacity", &automatic)) {
                edit("settings", 0, {{"auto_capacity", automatic}});
            }
            const auto &plan = listing.at("plan");
            if (plan.contains("error")) {
                ImGui::TextWrapped("Preview: %s",
                                   plan.at("error").get_ref<const std::string &>().c_str());
            } else {
                ImGui::TextWrapped("Up to %llu particles across enabled layers",
                                   static_cast<unsigned long long>(
                                       plan.at("particle_upper_bound").get<uint64_t>()));
                const bool gpu = plan.at("gpu_required").get<bool>();
                ImGui::TextDisabled(
                    "%s planning allowance: %.0f MiB", gpu ? "GPU" : "RAM",
                    plan.at(gpu ? "gpu_planning_bytes" : "host_planning_bytes").get<double>() /
                        (1024 * 1024));
                if (!plan.at("memory").at("gpu_measured").get<bool>() &&
                    plan.at("gpu_required").get<bool>()) {
                    ImGui::TextWrapped("GPU budget is unavailable; the allowance is an estimate.");
                }
                if (plan.at("memory").at("gpu_measured").get<bool>() &&
                    plan.at("gpu_required").get<bool>()) {
                    ImGui::TextDisabled("Available GPU budget: %.0f MiB",
                                        plan.at("memory").at("gpu_available_bytes").get<double>() /
                                            (1024 * 1024));
                }
                if (!plan.at("memory_check_passed").get<bool>()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.55f, 0.25f, 1));
                    ImGui::TextWrapped(
                        "%s", plan.at("memory_message").get_ref<const std::string &>().c_str());
                    ImGui::PopStyleColor();
                    if (!domain.matter_seed_memory_guard) {
                        ImGui::TextWrapped("Memory guard is disabled; this is advisory.");
                    }
                }
            }
            if (ImGui::Button("Generate particles from all layers", ImVec2(-1, 0))) {
                edit("apply", 0, nlohmann::json::object());
            }
            if (listing.at("pending_recipe_changes").get<bool>()) {
                ImGui::TextWrapped("Layer changes have not been generated yet.");
            }
        }
        if (!error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.3f, 1.0f));
            ImGui::TextWrapped("Seed operation failed: %s", error.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::TextWrapped(
            "Each layer is born once. Recipe edits leave live particles in place. "
            "Generate replaces existing particles with all enabled layers. "
            "If settling is enabled, the layers settle together before time starts.");
        if (listing.at("enabled").get<bool>()) {
            if (ImGui::Button("Edit domain instead of layer")) {
                selectMatterSeed(domain.name, 0);
            }
            bool settle = domain.fluid_seed_settle;
            if (ImGui::Checkbox("Settle layers before time starts", &settle)) {
                edit("settings", 0, {{"settle", settle}});
            }
            if (ImGui::TreeNode("Advanced seed settings")) {
                bool guard = domain.matter_seed_memory_guard;
                if (ImGui::Checkbox("Guard available memory", &guard)) {
                    edit("settings", 0, {{"memory_guard", guard}});
                }
                ImGui::TextWrapped(
                    "Planning reserves 30%% of available memory. The allowance "
                    "is conservative; actual allocations may differ. "
                    "This guard does not change grid resolution or material physics.");
                int ppc = domain.fluid_seed_particles_per_cell;
                if (ImGui::SliderInt("Continuum particles per voxel", &ppc, 1, 64)) {
                    edit("settings", 0, {{"particles_per_cell", ppc}});
                }
                uint64_t capacity = domain.fluid_max_particles;
                ImGui::BeginDisabled(domain.matter_seed_auto_capacity);
                if (ImGui::InputScalar("Domain particle capacity", ImGuiDataType_U64, &capacity)) {
                    const auto result = rtapi::setFluidMaxParticles(domain.name, capacity);
                    error = result.ok ? std::string() : result.error;
                }
                ImGui::EndDisabled();
                ImGui::TextWrapped("Capacity is a limit, not a target particle count. "
                                   "If all layers do not fit, generation is cancelled. "
                                   "Increase capacity, reduce layer volume, or lower continuum "
                                   "particles per voxel.");
                if (settle) {
                    int steps = domain.fluid_settle_max_steps;
                    float tolerance = domain.fluid_settle_displacement_m;
                    if (ImGui::DragInt("Settle step limit", &steps, 1.0f, 1, 4000)) {
                        edit("settings", 0, {{"settle_max_steps", steps}});
                    }
                    if (ImGui::DragFloat("Settle tolerance (m)", &tolerance, 0.00001f, 0.0f,
                                         1.0f)) {
                        edit("settings", 0, {{"settle_tolerance", tolerance}});
                    }
                }
                ImGui::TreePop();
            }
        }
        static std::map<std::string, uint64_t> lastSelected;
        const auto selected_id = selectedMatterSeed(domain);
        const bool selection_changed = lastSelected[domain.name] != selected_id;
        lastSelected[domain.name] = selected_id;
        for (const auto &row : listing.at("sources")) {
            const uint64_t id = row.at("id").get<uint64_t>();
            const auto key = std::to_string(id);
            ImGui::PushID(key.c_str());
            const bool selected = selectedMatterSeed(domain) == id;
            if (selected && selection_changed) {
                ImGui::SetNextItemOpen(true);
            }
            if (ImGui::TreeNode(row.at("name").get_ref<const std::string &>().c_str())) {
                if (ImGui::Button(selected ? "Selected in viewport" : "Edit in viewport")) {
                    selectMatterSeed(domain.name, id);
                }
                ImGui::TextWrapped("G: move, S: scale. Box layers stay axis-aligned. "
                                   "Generate applies edits to particles.");
                if (listing.at("plan").contains("layers")) {
                    for (const auto &estimate : listing.at("plan").at("layers")) {
                        if (estimate.at("id").get<uint64_t>() == id) {
                            ImGui::TextDisabled(
                                "Particle upper bound: %llu",
                                static_cast<unsigned long long>(
                                    estimate.at("particle_upper_bound").get<uint64_t>()));
                            break;
                        }
                    }
                }
                char name[64];
                std::snprintf(name, sizeof(name), "%s",
                              row.at("name").get_ref<const std::string &>().c_str());
                if (ImGui::InputText("Name", name, sizeof(name))) {
                    edit("update", id, {{"name", name}});
                }
                bool enabled = row.at("enabled").get<bool>();
                if (ImGui::Checkbox("Enabled", &enabled)) {
                    edit("update", id, {{"enabled", enabled}});
                }
                auto substance = row.at("substance").get<std::string>();
                if (SubstanceEditorUI::drawPicker("Substance", substance)) {
                    edit("update", id, {{"substance", substance}});
                }
                if (ImGui::TreeNode("Material settings")) {
                    ImGui::TextWrapped(
                        "Shared material edits also affect live users of that profile.");
                    if (SubstanceEditorUI::draw("SeedSourceMaterial", substance)) {
                        edit("update", id, {{"substance", substance}});
                    }
                    ImGui::TreePop();
                }
                const auto lo = row.at("bounds_min").get<std::vector<float>>();
                const auto hi = row.at("bounds_max").get<std::vector<float>>();
                float center[3], size[3];
                ImGui::BeginDisabled(!row.at("mesh_object").get_ref<const std::string &>().empty());
                for (int axis = 0; axis < 3; ++axis) {
                    center[axis] = lo[axis] + (hi[axis] - lo[axis]) * 0.5f;
                    size[axis] = hi[axis] - lo[axis];
                }
                bool moved = ImGui::DragFloat3("Box center (world)", center, 0.01f);
                moved |= ImGui::DragFloat3("Box size", size, 0.01f, 0.0001f, 10000.0f);
                if (moved) {
                    nlohmann::json low = nlohmann::json::array(), high = nlohmann::json::array();
                    for (int axis = 0; axis < 3; ++axis) {
                        low.push_back(center[axis] - size[axis] * 0.5f);
                        high.push_back(center[axis] + size[axis] * 0.5f);
                    }
                    edit("update", id, {{"bounds_min", low}, {"bounds_max", high}});
                }
                ImGui::EndDisabled();
                const auto mesh = row.at("mesh_object").get<std::string>();
                if (!mesh.empty() && ImGui::Button("Edit source mesh in viewport")) {
                    const auto result = rtapi::selectObject(mesh);
                    error = result.ok ? std::string() : result.error;
                }
                if (ImGui::BeginCombo("Volume", mesh.empty() ? "Box volume" : mesh.c_str())) {
                    if (ImGui::Selectable("Box volume", mesh.empty())) {
                        edit("update", id, {{"mesh_object", ""}});
                    }
                    for (const auto &object : rtapi::listObjects()) {
                        if (ImGui::Selectable(object.c_str(), object == mesh)) {
                            edit("update", id, {{"mesh_object", object}});
                        }
                    }
                    ImGui::EndCombo();
                }
                const char *modes[] = {"Material birth default", "Domain ambient", "Custom Kelvin"};
                const auto mode = row.at("temperature_mode").get<std::string>();
                int choice = mode == "custom" ? 2 : mode == "domain" ? 1 : 0;
                if (ImGui::Combo("Initial temperature", &choice, modes, 3)) {
                    edit("update", id,
                         {{"temperature_mode", choice == 2   ? "custom"
                                               : choice == 1 ? "domain"
                                                             : "material"}});
                }
                if (choice == 2) {
                    float kelvin = row.at("temperature_kelvin").get<float>();
                    char birth_fmt[48];
                    if (ImGui::DragFloat("Birth temperature", &kelvin, 0.5f, 1.0f, 3000.0f,
                            UiTemperature::kelvinFormat(kelvin, birth_fmt, sizeof(birth_fmt), "%.2f"))) {
                        edit("update", id, {{"temperature_kelvin", kelvin}});
                    }
                }
                char resolved_text[64];
                ImGui::Text("Resolved birth: %s", UiTemperature::kelvinText(
                    static_cast<float>(row.at("resolved_birth_kelvin").get<double>()),
                    resolved_text, sizeof(resolved_text), "%.2f"));
                auto velocity = row.at("velocity").get<std::vector<float>>();
                if (ImGui::DragFloat3("Initial velocity (m/s)", velocity.data(), 0.01f)) {
                    edit("update", id, {{"velocity", velocity}});
                }
                float wetness = row.at("grain_saturation").get<float>();
                const auto *profile = tryFindSubstance(substance);
                const bool dem_body = profile &&
                    profile->default_constitutive_model == MatterConstitutiveModel::Granular &&
                    profile->granular_transport == MatterGranularTransport::Dem;
                ImGui::BeginDisabled(!dem_body);
                auto bond_settings = row.at("bond_body");
                if (editDemBondSeed(bond_settings)) {
                    edit("update", id, {{"bond_body", bond_settings}});
                }
                ImGui::EndDisabled();
                const bool wet_grains =
                    profile &&
                    profile->default_constitutive_model == MatterConstitutiveModel::Granular &&
                    profile->granular_transport == MatterGranularTransport::Dem &&
                    profile->grain_water_capacity_fraction > 0.0f;
                ImGui::BeginDisabled(!wet_grains);
                if (ImGui::SliderFloat("DEM initial saturation", &wetness, 0.0f, 1.0f)) {
                    edit("update", id, {{"grain_saturation", wetness}});
                }
                ImGui::EndDisabled();
                if (ImGui::Button("Remove layer")) {
                    edit("remove", id, nlohmann::json::object());
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (listing.at("enabled").get<bool>()) {
            if (ImGui::Button("Clear particles and seed recipe", ImVec2(-1, 0))) {
                const auto cleared = rtapi::clearFluidParticles(domain.name, true);
                error = cleared.ok ? std::string() : cleared.error;
            }
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
}

} // namespace RayTrophiSim::Fluid
