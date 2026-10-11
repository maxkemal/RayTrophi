#include "UI/CloudMaterialUI.h"

#include "Api/RtApi.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "VDBVolume.h"
#include "Renderer.h"
#include "scene_ui.h"
#include "imgui.h"

#include <string>

namespace cloud_ui {

bool isWorldCloud(const VDBVolume* volume) {
    return volume && volume->name == "__RayTrophi_Internal_SkyCloudVolume";
}

static bool drawOpticalControls(atmosphere::CloudMaterial& m) {
    bool changed = false;
    if (!ImGui::TreeNodeEx("Cloud material", ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }
    changed |= ImGui::DragFloat("Density multiplier", &m.density_multiplier,
                               0.01f, 0.0f, 100.0f, "%.3f");
    changed |= ImGui::DragFloat("Scattering coefficient", &m.scattering_coefficient,
                               0.01f, 0.0f, 100.0f, "%.3f");
    changed |= ImGui::ColorEdit3("Scattering colour", &m.scattering_color.x,
                                ImGuiColorEditFlags_Float);
    changed |= ImGui::DragFloat("Absorption coefficient", &m.absorption_coefficient,
                               0.01f, 0.0f, 100.0f, "%.3f");
    ImGui::TextWrapped("Coefficients scale the layer extinction and cirrus optical depth. "
                       "Scattering colour is a linear RGB albedo tint.");
    changed |= ImGui::Checkbox("Override anisotropy", &m.override_anisotropy);
    ImGui::BeginDisabled(!m.override_anisotropy);
    changed |= ImGui::SliderFloat("Anisotropy", &m.anisotropy, -0.99f, 0.99f, "%.3f");
    ImGui::EndDisabled();
    ImGui::TextWrapped("Without an override, the droplet diameter controls the phase. "
                       "Cirrus uses its ice phase.");
    changed |= ImGui::ColorEdit3("Emission colour", &m.emission_color.x,
                                ImGuiColorEditFlags_Float);
    changed |= ImGui::DragFloat("Emission strength", &m.emission_strength,
                               0.01f, 0.0f, 100000.0f, "%.3f");
    ImGui::TextWrapped("Emission is source radiance per optical depth. Zero density or "
                       "zero total extinction disables the cloud material.");
    ImGui::TreePop();
    return changed;
}

bool drawMaterialControls(atmosphere::CloudState& clouds) {
    return drawOpticalControls(clouds.material);
}

bool drawLayerControls(atmosphere::CloudLayer& layer) {
    bool changed = false;
    changed |= ImGui::Combo("Formation role", &layer.formation_role,
                            "Generic\0Storm base\0Tower\0Anvil\0");
    changed |= ImGui::DragInt("Formation group", &layer.formation_group, 1.0f, 0, 3);
    ImGui::TextWrapped("Group 0 is independent. Matching groups share column centres "
                       "and their full vertical span. Match directions to keep them aligned.");
    changed |= ImGui::Checkbox("Layer optical modifiers", &layer.use_custom_material);
    if (layer.use_custom_material) {
        ImGui::TextWrapped("Density, scattering and colour multiply the common material. "
                           "Absorption and emission add to it. A local phase override wins.");
        changed |= drawOpticalControls(layer.material);
    }
    return changed;
}

void drawWorldCloudProperties(UIContext& ctx) {
    ImGui::PushID("WorldCloudProperties");
    ImGui::TextUnformatted("World clouds");
    ImGui::TextWrapped("This material is shared by the atmosphere cloud layers and cirrus. "
                       "Formation and weather controls are in World > Clouds.");
    ImGui::TextDisabled("Material transport: Vulkan RT and RayFusion.");
    atmosphere::CloudState clouds = ctx.renderer.world.getClouds();
    bool changed = drawMaterialControls(clouds);
    if (ImGui::TreeNodeEx("Layer optics", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
            ImGui::PushID(i);
            auto& layer = clouds.layers[i];
            ImGui::Text("Layer %d", i + 1);
            changed |= ImGui::Checkbox("Enabled", &layer.enabled);
            changed |= ImGui::DragFloat("Extinction (1/m)", &layer.extinction_per_m,
                                       0.001f, 0.0f, 2.0f, "%.4f");
            changed |= ImGui::SliderFloat("Droplet diameter (um)",
                                         &layer.droplet_diameter_um, 5.0f, 50.0f, "%.1f");
            changed |= ImGui::SliderFloat("Direction (deg)", &layer.rotation_degrees,
                                         -360.0f, 360.0f, "%.1f");
            changed |= drawLayerControls(layer);
            ImGui::PopID();
        }
        changed |= ImGui::Checkbox("Cirrus enabled", &clouds.cirrus.enabled);
        changed |= ImGui::DragFloat("Cirrus optical depth", &clouds.cirrus.opacity,
                                   0.01f, 0.0f, 5.0f, "%.3f");
        changed |= ImGui::SliderFloat("Cirrus direction (deg)",
                                     &clouds.cirrus.rotation_degrees,
                                     -360.0f, 360.0f, "%.1f");
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Quality")) {
        changed |= ImGui::DragInt("RT march steps", &clouds.quality.rt_steps, 1.0f, 16, 1024);
        changed |= ImGui::Checkbox("RT reference path tracer",
                                  &clouds.quality.rt_reference_path_trace);
        changed |= ImGui::DragInt("RT max bounces", &clouds.quality.rt_max_bounces,
                                 1.0f, 1, 1024);
        changed |= ImGui::DragInt("Realtime steps", &clouds.quality.realtime_steps,
                                 1.0f, 8, 512);
        ImGui::TreePop();
    }
    static std::string error;
    if (changed) {
        nlohmann::json patch;
        atmosphere::cloudsToJson(clouds, patch);
        const rtapi::Result result = rtapi::setCloudsJson(patch.dump());
        error = result.ok ? std::string{} : result.error;
    }
    if (!error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s", error.c_str());
    }
    ImGui::PopID();
}

} // namespace cloud_ui
