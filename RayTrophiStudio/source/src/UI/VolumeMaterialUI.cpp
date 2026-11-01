#include "UI/VolumeMaterialUI.h"
#include "VolumeMaterialAuthoring.h"
#include "VolumeMaterialService.h"
#include "Api/RtApi.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

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
                ImGui::TextWrapped("Material edits here belong to this source in domain '%s' "
                    "and persist when the simulation republishes.", domains[d].name.c_str());
                if (stream.merged_shader) {
                    ImGui::TextWrapped("This volume combines compatible gas sources. "
                        "You are editing the source shown above. Changing its color or "
                        "emission can separate the volumes and increase render cost.");
                }
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

void hint(const char* text) {
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", text);
}

bool applyPatch(VolumeShader& shader, const nlohmann::json& patch) {
    static std::string error;
    bool changed = false;
    if (!patch.empty()) changed = volume_material::configure(shader, patch, error);
    if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
    return changed;
}

template <typename Bins>
void drawHistogram(const char* id, const Bins& bins) {
    if (bins.empty()) return;
    std::vector<float> values(bins.begin(), bins.end());
    ImGui::PlotHistogram(id, values.data(), static_cast<int>(values.size()), 0, nullptr,
        0.0f, FLT_MAX, ImVec2(-1.0f, 48.0f * ImGui::GetFontSize() / 13.0f));
}

} // namespace

bool volumeShaderHasGasDomain(UIContext& context, const std::shared_ptr<VolumeShader>& shader) {
    for (const auto& candidate : context.scene.ensureParticleSimulationSystem().gridDomains()) {
        if (candidate.shader == shader) return true;
    }
    return false;
}

bool drawGasTemperatureFitUI(UIContext& context, const std::shared_ptr<VolumeShader>& shader) {
    std::string domain;
    for (const auto& candidate : context.scene.ensureParticleSimulationSystem().gridDomains()) {
        if (candidate.shader == shader) domain = candidate.name;
    }
    if (domain.empty()) return false;
    static rtapi::GasFieldStats stats;
    static std::string statsDomain;
    static std::string error;
    bool changed = false;
    if (ImGui::Button("Measure field")) {
        const auto result = rtapi::measureGasFields(domain, 0.01f, 64, stats);
        error = result.ok ? std::string() : result.error;
        statsDomain = domain;
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit temperature to field")) {
        rtapi::GasTemperatureFit fit;
        const auto result = rtapi::fitGasTemperatureRange(domain, {}, fit);
        error = result.ok ? std::string() : result.error;
        changed = result.ok;
    }
    hint("Fit writes the temperature range and gamma once from the current frame, so a "
        "sequence does not flicker. Use a representative frame.");
    if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
    if (statsDomain == domain && stats.temperature_measured) {
        const auto& t = stats.temperature_kelvin;
        ImGui::Text("Temperature K  p5 %.0f  p50 %.0f  p95 %.0f  peak %.0f", t.p5, t.p50, t.p95, t.maximum);
        if (stats.emission_measured) {
            ImGui::Text("Ramp position  p50 %.2f  p95 %.2f", stats.ramp_coordinate.p50,
                        stats.ramp_coordinate.p95);
            drawHistogram("##RampHistogram", stats.ramp_histogram);
            hint("Where the field lands on the fire ramp, 0 to 1. A narrow spike means "
                "the ramp is flattening differences: fit the range or change the gamma.");
            ImGui::Text("Over radiance limit: %.1f%% of emitting cells",
                stats.emitting_cells ? 100.0 * double(stats.limited_cells) /
                    double(stats.emitting_cells) : 0.0);
        }
        if (stats.flame_measured) {
            ImGui::Text("Burning cells: %zu of %zu", stats.burning_cells, stats.active_cells);
        }
    }
    return changed;
}

bool drawVolumePresetsUI(VolumeShader& shader) {
    nlohmann::json patch = nlohmann::json::object();
    if (ImGui::Button("Physical Cloud")) {
        patch["preset"] = "physical_cloud";
    }
    hint("External lighting, low absorption, emission off. Density scale, source "
        "channels and temperature range are preserved.");
    ImGui::SameLine();
    if (ImGui::Button("Blackbody Fire")) {
        patch["preset"] = "blackbody_fire";
    }
    ImGui::SameLine();
    if (ImGui::Button("Artistic Fire Ramp")) {
        patch["preset"] = "artistic_fire";
    }
    if (!shader.material_graph.empty()) {
        ImGui::TextWrapped("A Volume Graph is active: its outputs may override these values "
            "and disable the multiple-scattering walk.");
    }
    return applyPatch(shader, patch);
}

bool drawSmokeAppearanceUI(VolumeShader& shader) {
    const auto look = volume_material::smokeAppearance(shader);
    nlohmann::json patch = nlohmann::json::object();
    float color[3] = {look.color.x, look.color.y, look.color.z};
    if (ImGui::ColorEdit3("Color", color)) {
        patch["smoke_color"] = {color[0], color[1], color[2]};
    }
    hint("Color of lit smoke. Dark gray is soot, near white is steam. Does not change "
        "how dense the smoke looks.");
    float opacity = look.opacity;
    if (ImGui::DragFloat("Opacity", &opacity, 0.01f, 0.0f, 10000.0f, "%.3f")) {
        patch["smoke_opacity"] = (std::max)(opacity, 0.0f);
    }
    hint("How much light the smoke stops. Does not change its color.");
    float tint[3] = {look.tint.x, look.tint.y, look.tint.z};
    if (ImGui::ColorEdit3("Depth Tint", tint)) {
        patch["smoke_tint"] = {tint[0], tint[1], tint[2]};
    }
    hint("Which colors thick smoke removes. White is neutral.");
    return applyPatch(shader, patch);
}

bool drawVolumeDensityRampUI(VolumeShader& shader) {
    nlohmann::json patch = nlohmann::json::object();
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
    return applyPatch(shader, patch);
}

bool drawFireLightingUI(VolumeShader& shader) {
    nlohmann::json patch = nlohmann::json::object();
    int internalSteps = shader.quality.internal_light_steps;
    if (ImGui::SliderInt("Internal Fire Light Steps", &internalSteps, 0, 32)) {
        patch["internal_light_steps"] = internalSteps;
    }
    hint("Fire lights the smoke around it, on a fixed budget per camera sample, so it "
        "stays clean and cheap where the random walk is noisy. 0 is off. RT samples "
        "scattering directions; RayFusion uses six fixed directions.");
    ImGui::BeginDisabled(internalSteps <= 0);
    int internalStride = shader.quality.internal_light_stride;
    if (ImGui::SliderInt("Internal Light Update Stride", &internalStride, 1, 32)) {
        patch["internal_light_stride"] = internalStride;
    }
    hint("Camera samples between two gathers. 1 is the smoothest and the most expensive.");
    ImGui::EndDisabled();
    return applyPatch(shader, patch);
}

bool drawFireResponseUI(VolumeShader& shader) {
    nlohmann::json patch = nlohmann::json::object();
    const auto scalar = [&](const char* label, const char* key, float value,
                            float lo, float hi, const char* text) {
        if (ImGui::SliderFloat(label, &value, lo, hi, "%.3f")) {
            patch[key] = value;
        }
        hint(text);
    };
    ImGui::SeparatorText("Response");
    bool independent = shader.emission_response.independent_emission;
    if (ImGui::Checkbox("Emission independent of smoke", &independent)) {
        patch["independent_emission"] = independent;
    }
    hint("Fire brightness stops following the smoke curve and multiplier; smoke still "
        "absorbs the light. Needed for soot burnout. Existing intensity may need retuning.");
    bool enabled = shader.emission_response.enabled;
    if (ImGui::Checkbox("Emission response", &enabled)) {
        patch["response_enabled"] = enabled;
    }
    hint("Off restores the plain emission equation.");
    ImGui::BeginDisabled(!enabled);
    scalar("Temperature intensity power", "temperature_power",
        shader.emission_response.temperature_power, 0.0f, 8.0f,
        "Shapes brightness separately from ramp color.");
    scalar("Emission density power", "density_power",
        shader.emission_response.density_power, 0.0f, 4.0f,
        "0 keeps thin cells as bright as dense ones. Empty cells stay dark.");
    scalar("Emission exposure (EV)", "emission_exposure_ev",
        shader.emission_response.exposure_ev, -16.0f, 16.0f, "Brightness of all fire, in stops.");
    scalar("Emission saturation", "emission_saturation",
        shader.emission_response.saturation, 0.0f, 2.0f, "Color strength of the fire.");
    ImGui::EndDisabled();
    scalar("Temperature gamma", "temperature_gamma",
        shader.emission_response.temperature_gamma, 0.1f, 8.0f,
        "Moves the field along the ramp: above 1 spreads cool gas, below 1 spreads hot gas.");
    float flameEmission = shader.emission_response.flame_emission;
    if (ImGui::DragFloat("Flame emission", &flameEmission, 0.01f, 0.0f, 100000.0f, "%.3f")) {
        patch["flame_emission"] = (std::max)(flameEmission, 0.0f);
    }
    hint("Adds light from the combustion front itself, on top of the temperature glow. "
        "Works where the gas is too thin or too cool to glow on its own. Live simulation: "
        "a cached or imported volume has no flame field.");

    ImGui::SeparatorText("Soot");
    ImGui::BeginDisabled(!independent);
    float sootBurnout = shader.emission_response.soot_burnout;
    if (ImGui::SliderFloat("Soot burnout", &sootBurnout, 0.0f, 1.0f)) {
        patch["soot_burnout"] = sootBurnout;
    }
    hint("Hot gas holds less smoke, so a fireball is seen through its own smoke. Fades "
        "between start and end on the ramp's 0..1 temperature. Shadows ignore it. "
        "Needs emission independent of smoke.");
    float sootStart = shader.emission_response.soot_burnout_start;
    float sootEnd = shader.emission_response.soot_burnout_end;
    if (ImGui::SliderFloat("Burnout start", &sootStart, 0.0f, 0.99f)) {
        patch["soot_burnout_start"] = sootStart;
        patch["soot_burnout_end"] = (std::max)(sootEnd, sootStart + 0.01f);
    }
    if (ImGui::SliderFloat("Burnout end", &sootEnd, 0.01f, 1.0f)) {
        patch["soot_burnout_end"] = sootEnd;
        patch["soot_burnout_start"] = (std::min)(sootStart, sootEnd - 0.01f);
    }
    ImGui::EndDisabled();

    if (ImGui::TreeNode("Advanced Fire")) {
        int temperatureUnits = shader.emission_response.temperature_units;
        const char* unitNames[] = {"Source default", "Kelvin", "Normalized range"};
        if (ImGui::Combo("Temperature units", &temperatureUnits, unitNames, 3)) {
            patch["temperature_units"] = temperatureUnits;
        }
        hint("Source default uses Kelvin for live simulation volumes and legacy inference "
            "for imported VDBs. Explicit units keep cold or missing fields dark.");
        bool maskEnabled = shader.emission_response.mask_enabled;
        if (ImGui::Checkbox("Density emission mask", &maskEnabled)) {
            patch["emission_mask_enabled"] = maskEnabled;
        }
        hint("Dense smoke suppresses emission while keeping smoke opacity. Masked gas "
            "sources render separately, so overlapping sources may cost more.");
        ImGui::BeginDisabled(!maskEnabled);
        float maskCenter = shader.emission_response.mask_center;
        float maskWidth = shader.emission_response.mask_width;
        float maskStrength = shader.emission_response.mask_strength;
        if (ImGui::DragFloat("Mask center", &maskCenter, 0.005f, 0.0f, 100000.0f, "%.4f")) {
            patch["emission_mask_center"] = maskCenter;
        }
        hint("Higher reveals more fire. Uses evaluated density before its multiplier.");
        if (ImGui::DragFloat("Mask width", &maskWidth, 0.005f, 1e-6f, 100000.0f, "%.4f")) {
            patch["emission_mask_width"] = maskWidth;
        }
        if (ImGui::SliderFloat("Mask strength", &maskStrength, 0.0f, 1.0f, "%.3f")) {
            patch["emission_mask_strength"] = maskStrength;
        }
        ImGui::EndDisabled();
        ImGui::TreePop();
    }
    return applyPatch(shader, patch);
}
