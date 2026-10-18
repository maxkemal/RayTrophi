#include "UI/AudioAcousticSettingsUI.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <utility>

namespace AudioUI {
void drawSceneAcousticSettings(UIContext& ctx) {
    using json = nlohmann::json;
    if (!ImGui::CollapsingHeader("Scene acoustics", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    const auto settings = ctx.scene.audio->invoke("acoustics_get", json::object());
    const auto& config = settings.at("config");
    ImGui::TextWrapped("Saved with the scene. New spatial sounds follow these defaults.");
    ImGui::TextWrapped("Sources with custom settings or Off keep their own choice.");
    for (const auto& field : {std::pair{"enabled", "Enable scene acoustics"},
        std::pair{"spectral", "Frequency dependent transmission"},
        std::pair{"reflections", "Early reflections"},
        std::pair{"diffuse", "Diffuse scattering"}}) {
        bool value = config.at(field.first);
        if (ImGui::Checkbox(field.second, &value)) {
            workspaceCall(ctx, "acoustics_update", {{"config", {{field.first, value}}}});
        }
    }
    if (ImGui::TreeNode("Scene scale and geometry")) {
        float units = config.at("meters_per_unit");
        if (slider("Meters per scene unit", units, 0.001f, 1000.0f, "%.3f m",
            ImGuiSliderFlags_Logarithmic)) {
            workspaceCall(ctx, "acoustics_update", {{"config", {{"meters_per_unit", units}}}});
        }
        int mode = config.at("boundary_mode") == "object_barrier" ? 0 : 1;
        if (ImGui::Combo(property("Surface interpretation").c_str(), &mode,
            "One barrier per object\0Independent thin surfaces\0")) {
            workspaceCall(ctx, "acoustics_update", {{"config", {{"boundary_mode",
                mode == 0 ? "object_barrier" : "thin_surface"}}}});
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Source exceptions")) {
        int exceptions = 0;
        int nonSpatial = 0;
        for (const auto& source : settings.at("sources")) {
            exceptions += source.at("policy").at("mode") != "scene" ? 1 : 0;
            nonSpatial += source.at("spatial_eligible").get<bool>() ? 0 : 1;
        }
        ImGui::Text("%d exceptions / %d non-spatial sounds", exceptions, nonSpatial);
        ImGui::TextWrapped("Reset removes custom/off choices from every source. "
            "Non-spatial sounds remain outside geometry acoustics.");
        ImGui::BeginDisabled(exceptions == 0);
        if (ImGui::Button("Use scene settings for all sources")) {
            workspaceCall(ctx, "acoustics_reset_sources", {{"scope", "scene"}});
        }
        ImGui::EndDisabled();
        ImGui::TreePop();
    }
}
}
