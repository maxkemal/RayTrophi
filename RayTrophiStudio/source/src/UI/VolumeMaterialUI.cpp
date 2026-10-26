#include "UI/VolumeMaterialUI.h"
#include "VolumeMaterialAuthoring.h"
#include "VolumeMaterialService.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>

bool drawVolumeAdaptiveStepping(VolumeShader& shader) {
    bool adaptive = shader.quality.adaptive_stepping;
    if (!ImGui::Checkbox("Adaptive Volume Steps", &adaptive)) {
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("RT mixtures adapt to all density fields and stop at voxel "
                "boundaries. Noise/graphs use fixed steps; emissive fields do not coarsen. "
                "Voxels / Sample remains the base resolution.");
        }
        return false;
    }
    std::string error;
    return volume_material::configure(shader, {{"adaptive_stepping", adaptive}}, error);
}

void drawMatterFogLinks(UIContext& context, SceneUI& ui,
    const RayTrophiSim::SimulationGridDomainDesc& domain) {
    if (domain.type != RayTrophiSim::SimulationDomainType::Matter) {
        return;
    }
    for (const auto& system : context.scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        const auto& domains = system.runtime->gridDomains();
        for (std::size_t d = 0; d < domains.size() &&
                d < system.matter_fog_streams.size(); ++d) {
            if (&domains[d] != &domain) {
                continue;
            }
            for (const auto& [key, stream] : system.matter_fog_streams[d]) {
                if (!stream.volume) {
                    continue;
                }
                ImGui::PushID(RayTrophiSim::Fluid::fogFieldId(key).c_str());
                if (ImGui::SmallButton(("Edit " +
                        stream.volume->name + " Fog...").c_str())) {
                    context.selection.selectVDBVolume(stream.volume, -1, stream.volume->name);
                    ui.tab_to_focus = "VDB";
                }
                ImGui::PopID();
            }
        }
    }
}

bool drawMatterFogVisibility(UIContext& context, VDBVolume* volume) {
    for (auto& system : context.scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        auto& domains = system.runtime->gridDomains();
        for (std::size_t d = 0; d < system.matter_fog_streams.size() &&
                d < domains.size(); ++d) {
            for (const auto& [key, stream] : system.matter_fog_streams[d]) {
                if (stream.volume.get() != volume) {
                    continue;
                }
                auto& appearance = domains[d].matter_fog_streams.at(key);
                bool visible = appearance.visible;
                ImGui::TextDisabled("Live Matter field: %s",
                    volume->name.c_str());
                if (ImGui::Checkbox("Visible##MatterFog", &visible)) {
                    const auto result = volume_material::execute(context, "set",
                        "matter:" + domains[d].name + ":" +
                            RayTrophiSim::Fluid::fogFieldId(key), {{"visible", visible}});
                    return !result.contains("__error");
                }
                return false;
            }
        }
    }
    return false;
}

namespace {

bool drawDensityProfile(VolumeDensityCurve& curve) {
    const float scale = ImGui::GetFontSize() / 13.0f;
    const ImVec2 size(std::max(ImGui::GetContentRegionAvail().x, 40.0f), 105.0f * scale);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##DensityProfile", size);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    const bool activated = ImGui::IsItemActivated();
    const float pad = 7.0f * scale;
    const ImVec2 lo(origin.x + pad, origin.y + pad);
    const ImVec2 hi(origin.x + size.x - pad, origin.y + size.y - pad);
    const float width = std::max(hi.x - lo.x, 1.0f);
    const float height = std::max(hi.y - lo.y, 1.0f);
    const auto clamp01 = [](float value) {
        return std::clamp(value, 0.0f, 1.0f);
    };
    bool changed = false;
    if (active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const auto& io = ImGui::GetIO();
        const float x = clamp01((io.MousePos.x - lo.x) / width);
        const float y = clamp01((hi.y - io.MousePos.y) / height);
        const float previousX = activated ? x :
            clamp01((io.MousePos.x - io.MouseDelta.x - lo.x) / width);
        const float previousY = activated ? y :
            clamp01((hi.y - io.MousePos.y + io.MouseDelta.y) / height);
        const int nearest = std::clamp(int(std::round(x * 7.0f)), 1, 7);
        for (int i = 1; i < 8; ++i) {
            const float knot = float(i) / 7.0f;
            const bool crossed = knot >= std::min(previousX, x) &&
                knot <= std::max(previousX, x);
            if (i != nearest && !crossed) {
                continue;
            }
            float value = y;
            if (crossed && std::abs(x - previousX) > 1e-6f) {
                const float t = (knot - previousX) / (x - previousX);
                value = previousY + (y - previousY) * t;
            }
            changed |= curve.values[i] != value;
            curve.values[i] = value;
            // Move neighbouring values with the brush to retain a monotone profile.
            for (int j = 1; j < i; ++j) {
                curve.values[j] = std::min(curve.values[j], value);
            }
            for (int j = i + 1; j < 8; ++j) {
                curve.values[j] = std::max(curve.values[j], value);
            }
        }
    }
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y),
        ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f * scale);
    const ImU32 grid = ImGui::GetColorU32(ImGuiCol_Border);
    for (int i = 0; i <= 4; ++i) {
        const float t = float(i) / 4.0f;
        draw->AddLine(ImVec2(lo.x + width * t, lo.y),
            ImVec2(lo.x + width * t, hi.y), grid);
        draw->AddLine(ImVec2(lo.x, lo.y + height * t),
            ImVec2(hi.x, lo.y + height * t), grid);
    }
    const ImU32 line = ImGui::GetColorU32(
        active ? ImGuiCol_PlotLinesHovered : ImGuiCol_PlotLines);
    ImVec2 previous(lo.x, hi.y);
    for (int i = 0; i < 8; ++i) {
        const ImVec2 point(lo.x + width * float(i) / 7.0f,
            hi.y - height * curve.values[i]);
        if (i > 0) {
            draw->AddLine(previous, point, line, 2.0f * scale);
        }
        draw->AddCircleFilled(point, 2.5f * scale, line);
        previous = point;
    }
    if (hovered || active) {
        ImGui::SetTooltip("Drag to paint density profile. X: input, Y: output.\n"
            "Zero stays empty; the profile remains increasing.");
    }
    return changed;
}

} // namespace

bool drawPhysicalVolumeMaterialUI(VolumeShader& shader) {
    if (!ImGui::TreeNodeEx("Physical volume / fire response", ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }
    nlohmann::json patch = nlohmann::json::object();
    if (ImGui::Button("Physical Cloud")) {
        patch["preset"] = "physical_cloud";
    }
    ImGui::SameLine();
    if (ImGui::Button("Blackbody Fire")) {
        patch["preset"] = "blackbody_fire";
    }
    ImGui::SameLine();
    if (ImGui::Button("Artistic Fire Ramp")) {
        patch["preset"] = "artistic_fire";
    }
    ImGui::TextWrapped("Cloud: external lighting, low absorption, emission off. "
        "Vulkan RT uses multiple scattering; RayFusion uses an approximation. "
        "Density scale, source channels and temperature range are preserved.");
    if (!shader.material_graph.empty()) {
        ImGui::TextWrapped("A Volume Graph is active: its outputs may override material "
            "values and disable the multiple-scattering walk.");
    }
    bool densityEnabled = shader.density.curve.enabled;
    auto densityCurve = shader.density.curve;
    bool densityChanged = ImGui::Checkbox("Density ramp", &densityEnabled);
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset##DensityRamp")) {
        densityCurve = VolumeDensityCurve{};
        densityChanged = true;
    }
    densityCurve.enabled = densityEnabled;
    ImGui::BeginDisabled(!densityEnabled);
    densityChanged |= drawDensityProfile(densityCurve);
    ImGui::EndDisabled();
    if (densityChanged) {
        patch["density_curve"] = densityCurve.toJson();
    }
    bool enabled = shader.emission_response.enabled;
    if (ImGui::Checkbox("Independent emission response", &enabled)) {
        patch["response_enabled"] = enabled;
    }
    ImGui::BeginDisabled(!enabled);
    const auto scalar = [&](const char* label, const char* key, float value,
                            float lo, float hi) {
        if (ImGui::SliderFloat(label, &value, lo, hi, "%.3f")) {
            patch[key] = value;
        }
    };
    scalar("Temperature intensity power", "temperature_power",
        shader.emission_response.temperature_power, 0.0f, 8.0f);
    scalar("Emission density power", "density_power",
        shader.emission_response.density_power, 0.0f, 4.0f);
    scalar("Emission exposure (EV)", "emission_exposure_ev",
        shader.emission_response.exposure_ev, -16.0f, 16.0f);
    scalar("Emission saturation", "emission_saturation",
        shader.emission_response.saturation, 0.0f, 2.0f);
    ImGui::EndDisabled();
    int internalSteps = shader.quality.internal_light_steps;
    if (ImGui::SliderInt("Internal fire light steps (0=off)", &internalSteps, 0, 32)) {
        patch["internal_light_steps"] = internalSteps;
    }
    int internalStride = shader.quality.internal_light_stride;
    if (ImGui::SliderInt("Internal light update stride", &internalStride, 1, 32)) {
        patch["internal_light_stride"] = internalStride;
    }
    ImGui::TextWrapped("Internal light gathers emission through this volume into smoke. "
        "RT samples scattering directions; RayFusion uses six fixed directions. "
        "More steps and stride 1 improve sampling at additional cost.");
    ImGui::TextWrapped("Temperature power shapes intensity separately from ramp colour. "
        "Density power 0 keeps positive thin cells luminous; empty cells stay dark. "
        "Response off restores the legacy emission equation. The radiance limiter remains.");
    if (shader.scattering.random_walk) {
        ImGui::TextWrapped("RT mixtures support overlapping contributors and independent "
            "emission. Density noise and spatial Volume Graph fields retain fixed steps. "
            "Dense clouds need sufficient event and sample budgets.");
    }
    static std::string error;
    bool changed = false;
    if (!patch.empty()) {
        changed = volume_material::configure(shader, patch, error);
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::TreePop();
    return changed;
}
