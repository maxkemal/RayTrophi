#include "UI/CloudMaterialUI.h"

#include "Api/RtApi.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "Backend/IBackend.h"
#include "Backend/CloudLightningParams.h"
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

static bool drawLightningControls(atmosphere::CloudState& clouds) {
    bool changed = false;
    if (ImGui::TreeNodeEx("Lightning / Storm flashes", ImGuiTreeNodeFlags_DefaultOpen)) {
        atmosphere::CloudLightning& lightning = clouds.lightning;
        changed |= ImGui::Checkbox("Enable lightning", &lightning.enabled);
        changed |= ImGui::Checkbox("Lightning rate from climate",
                                   &lightning.derive_rate_from_climate);
        ImGui::BeginDisabled(lightning.derive_rate_from_climate);
        changed |= ImGui::SliderFloat("Flashes per minute", &lightning.rate_per_minute,
                                      0.0f, 600.0f, "%.1f");
        ImGui::EndDisabled();
        changed |= ImGui::SliderFloat("Flash duration (s)", &lightning.duration_s,
                                      0.08f, 3.0f, "%.2f");
        changed |= ImGui::DragFloat("Horizontal flash span (m)", &lightning.horizontal_span_m,
                                    100.0f, 100.0f, 50000.0f, "%.0f");
        changed |= ImGui::SliderFloat("Branching", &lightning.branching, 0.0f, 1.0f, "%.2f");
        const float effectiveRate = Backend::lightningEffectiveRate(lightning);
        if (effectiveRate + 0.1f < lightning.rate_per_minute) {
            ImGui::TextWrapped("At this duration: up to %.0f flashes/min. "
                               "Shorter duration allows more frequent flashes.", effectiveRate);
        }
        changed |= ImGui::DragFloat("Flash intensity", &lightning.intensity,
                                    0.1f, 0.0f, 10000.0f, "%.1f");
        changed |= ImGui::ColorEdit3("Flash colour", &lightning.color.x,
                                     ImGuiColorEditFlags_Float);
        float centerXZ[2] = {lightning.center_m.x, lightning.center_m.z};
        if (ImGui::DragFloat2("Storm centre X/Z (m)", centerXZ,
                             10.0f, -1000000.0f, 1000000.0f, "%.0f")) {
            lightning.center_m.x = centerXZ[0];
            lightning.center_m.z = centerXZ[1];
            changed = true;
        }
        changed |= ImGui::DragFloat("Storm extent (m)", &lightning.extent_m,
                                    100.0f, 100.0f, 200000.0f, "%.0f");
        changed |= ImGui::DragFloat("Strike ground altitude (m)", &lightning.ground_altitude_m,
                                    10.0f, -500.0f, 20000.0f, "%.0f");
        changed |= ImGui::SliderFloat("Ground strike fraction", &lightning.ground_strike_fraction,
                                      0.0f, 1.0f, "%.2f");
        ImGui::TextWrapped("Lower ground strike fraction produces more wide, horizontal "
                           "cloud-to-cloud flashes. Longer flashes develop their branches over time.");
        changed |= ImGui::DragFloat("Bolt radius (m)", &lightning.bolt_radius_m,
                                    0.1f, 0.1f, 30.0f, "%.1f");
        ImGui::TextWrapped("Vulkan RT and RayFusion share seeded timeline flashes. "
                           "Requires a convective cloud layer. Use timeline Play to see flashes. "
                           "Realtime Weather Preview animates rain/snow, not lightning. "
                           "Try 30-60 flashes/min while adjusting the storm footprint; "
                           "events outside the camera or clear cloud areas may remain invisible.");
        ImGui::TreePop();
    }
    return changed;
}

bool drawMaterialControls(atmosphere::CloudState& clouds) {
    return drawOpticalControls(clouds.material);
}

bool drawWeatherLightningControls(UIContext& ctx) {
    atmosphere::CloudState clouds = ctx.renderer.world.getClouds();
    const bool changed = drawLightningControls(clouds);
    static std::string error;
    if (changed) {
        nlohmann::json json;
        atmosphere::cloudsToJson(clouds, json);
        const rtapi::Result result = rtapi::setCloudsJson(
            nlohmann::json{{"lightning", json["lightning"]}}.dump());
        error = result.error;
        if (!result.ok) {
            ImGui::TextWrapped("%s", error.c_str());
            return false;
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    return changed;
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

void beginLegacyShaftControls(const UIContext& ctx, const atmosphere::CloudState& clouds) {
    const auto backend = ctx.backend_ptr ? ctx.backend_ptr->getInfo().type :
                                          Backend::BackendType::AUTO;
    const bool vulkan = backend == Backend::BackendType::VULKAN_RT ||
                        backend == Backend::BackendType::VULKAN_COMPUTE;
    bool naturalShafts = false;
    for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
        const auto& layer = clouds.layers[i];
        const auto material = atmosphere::layerMaterial(clouds, i);
        naturalShafts |= layer.enabled && layer.coverage > 0.0f &&
            layer.extinction_per_m > 0.0f && material.density_multiplier > 0.0f &&
            material.scattering_coefficient + material.absorption_coefficient > 0.0f;
    }
    naturalShafts &= vulkan;
    if (naturalShafts) {
        ImGui::TextWrapped("Vulkan cloud shafts come from the cloud lighting. "
                           "Sun direction, cloud height and gaps shape them in RT and RayFusion. "
                           "The extra legacy RT shaft pass is skipped while liquid clouds render.");
    }
    ImGui::BeginDisabled(naturalShafts);
}

void drawWorldCloudProperties(UIContext& ctx) {
    ImGui::PushID("WorldCloudProperties");
    ImGui::TextUnformatted("World clouds");
    ImGui::TextWrapped("This material is shared by the atmosphere cloud layers and cirrus. "
                       "Formation and weather controls are in World > Clouds.");
    ImGui::TextDisabled("Material transport: Vulkan RT and RayFusion.");
    atmosphere::CloudState clouds = ctx.renderer.world.getClouds();
    bool changed = drawMaterialControls(clouds);
    changed |= drawLightningControls(clouds);
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
