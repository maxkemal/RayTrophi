#include "DomainCreationUI.h"
#include "Api/DomainCreationApi.h"
#include "imgui.h"
#include <algorithm>
#include <string>

void drawDomainCreationActions(const std::function<void()> &reset, int &selected_domain_index) {
    static std::string error;
    const auto source = rtapi::selectedFlatDomainSource();
    const float width = std::max(60.0f, (ImGui::GetContentRegionAvail().x - 10.0f) / 3.0f);
    const auto create = [&](const std::string &object) {
        nlohmann::json result;
        const auto status = rtapi::createPlacedMatterDomain("", object, result);
        error = status.ok ? std::string() : status.error;
        if (status.ok) {
            selected_domain_index = result.at("domain_index").get<int>();
        }
    };
    if (ImGui::Button("Reset Sim##SimResetTop", ImVec2(width, 24))) {
        reset();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Clear the bake cache and return to live free-run preview.");
    }
    ImGui::SameLine();
    if (ImGui::Button("+ Add Domain##SimulationPanel", ImVec2(width, 24))) {
        create("");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Create a 5 m domain on the ground at the world origin. "
                          "Selection does not change placement.");
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(source.empty());
    if (ImGui::Button("+ From Select##SimulationPanel", ImVec2(width, 24))) {
        create(source);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fit a manual domain to the selected flat mesh's current world bounds.");
    }
    ImGui::EndDisabled();
    if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.3f, 1));
        ImGui::TextWrapped("Domain creation failed: %s", error.c_str());
        ImGui::PopStyleColor();
    }
}
