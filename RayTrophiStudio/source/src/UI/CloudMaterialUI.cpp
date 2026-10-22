#include "UI/CloudMaterialUI.h"
#include "UI/StormGizmoUI.h"

#include "Api/RtApi.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "Atmosphere/AtmosphereLocalWeather.h"
#include "Backend/IBackend.h"
#include "Backend/CloudLightningParams.h"
#include "VDBVolume.h"
#include "Renderer.h"
#include "scene_ui.h"
#include "imgui.h"

#include <cstdio>
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
    changed |= ImGui::SliderFloat("Scattering saturation", &m.scattering_saturation,
                                  0.0f, 2.0f, "%.2f");
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
    changed |= ImGui::SliderFloat("Emission saturation", &m.emission_saturation,
                                  0.0f, 2.0f, "%.2f");
    changed |= ImGui::SliderFloat("Emission exposure (EV)", &m.emission_exposure_ev,
                                  -16.0f, 16.0f, "%.2f");
    ImGui::TextWrapped("Saturation 0 gives grey; 1 keeps the authored colour. "
                       "Emission exposure +1 EV doubles emission only, not sunlight.");
    ImGui::TextWrapped("Emission is source radiance per optical depth. Zero density or "
                       "zero total extinction disables the cloud material.");
    ImGui::TreePop();
    return changed;
}

// Storm cells (docs/dev/ATMOSPHERE_STORMS.md): where it rains and flashes.
static bool freezeRequested = false;
static bool drawStormControls(atmosphere::CloudState& clouds) {
    bool changed = false;
    if (!ImGui::TreeNodeEx("Storms / convective cells", ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }
    atmosphere::CloudStorms& s = clouds.storms;
    changed |= ImGui::Checkbox("Storms from climate", &s.derive_from_climate);
    ImGui::BeginDisabled(s.derive_from_climate);
    changed |= ImGui::Checkbox("Enable storm cells", &s.enabled);
    changed |= ImGui::SliderInt("Concurrent cells", &s.max_cells, 0, atmosphere::kMaxAutoStormCells);
    changed |= ImGui::SliderFloat("Peak intensity", &s.intensity, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Stratiform rain", &s.stratiform_rain, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();
    float centerXZ[2] = {s.center_m.x, s.center_m.z};
    if (ImGui::DragFloat2("Storm domain centre X/Z (m)", centerXZ, 10.0f, -1000000.0f, 1000000.0f, "%.0f")) {
        s.center_m.x = centerXZ[0];
        s.center_m.z = centerXZ[1];
        changed = true;
    }
    changed |= ImGui::DragFloat("Storm domain extent (m)", &s.extent_m, 100.0f, 1000.0f, 500000.0f, "%.0f");
    changed |= ImGui::DragFloat("Cell radius (m)", &s.cell_radius_m, 10.0f, 500.0f, 20000.0f, "%.0f");
    changed |= ImGui::DragFloat("Cell lifetime (s)", &s.lifetime_s, 10.0f, 120.0f, 14400.0f, "%.0f");
    changed |= ImGui::SliderFloat("Updraft turrets", &s.turrets, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Rotation", &s.rotation, -1.0f, 1.0f, "%.2f");
    changed |= ImGui::Checkbox("Storm column (cloud base to tropopause)", &s.column);
    ImGui::BeginDisabled(!s.column);
    changed |= ImGui::DragFloat("Tropopause / anvil altitude (m)", &s.column_top_m,
                                10.0f, 2000.0f, 20000.0f, "%.0f");
    changed |= ImGui::SliderFloat("Anvil spread", &s.anvil, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Wall cloud", &s.wall_cloud, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Mammatus", &s.mammatus, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();
    ImGui::TextWrapped("Storm column: each cell is its own cloud from the lowest layer's base "
                       "to the tropopause (towers, overshooting tops, anvil downwind, wall cloud, "
                       "mammatus); the layers step aside inside the cells. Off: the cells only "
                       "raise the cover of the layers they sit in.");
    // Storm cell volumes (docs/dev/ATMOSPHERE_CLOUD_VOLUMES.md).
    ImGui::BeginDisabled(!s.column);
    atmosphere::CloudVolumes& v = clouds.volumes;
    changed |= ImGui::Checkbox("Cell volumes (3D signed distance)", &v.enabled);
    ImGui::BeginDisabled(!v.enabled);
    changed |= ImGui::SliderInt("Volume slots", &v.slots, 1, atmosphere::kMaxCloudVolumeSlots);
    changed |= ImGui::DragFloat("Voxel size (m)", &v.voxel_m, 1.0f, 20.0f, 400.0f, "%.0f");
    changed |= ImGui::DragFloat("Surface skin (m)", &v.skin_m, 1.0f, 5.0f, 500.0f, "%.0f");
    changed |= ImGui::SliderFloat("Min development", &v.min_tower, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    {
        const atmosphere::ColumnLevels lv = atmosphere::columnLevels(clouds);
        if (lv.valid) {
            ImGui::Text("Column: LCL %.0f m, freezing %.0f m, tropopause %.0f m",
                        lv.lcl_m, lv.freezing_m, lv.tropopause_m);
        }
    }
    ImGui::TextWrapped("Cell volumes: developed cells are drawn from a generated 3D volume "
                       "(%d m wide box, 16 MB per slot): thermal bubble towers, flat base with a "
                       "wall cloud, anvil at the tropopause. The rest keep the column surface.",
                       static_cast<int>(v.voxel_m * atmosphere::kCloudVolumeVoxels));
    ImGui::Checkbox("Show cells in viewport", &StormGizmoUI::visible);
    ImGui::TextWrapped("Cells grow, mature (heavy rain, lightning) and dissipate (light, wide "
                       "rain), drifting with the climate wind. Stratiform rain = broad light "
                       "rain under dense cover outside the cells.");
    ImGui::TextWrapped("Automatic cells cannot be edited one by one: convert the live ones "
                       "into authored cells (the sky stays the same, the automatic lanes stop), "
                       "or click a cell in the viewport.");
    if (ImGui::Button("Convert live automatic cells to authored")) freezeRequested = true;
    char authoredLabel[64];
    std::snprintf(authoredLabel, sizeof(authoredLabel), "Authored cells (%d)###AuthoredCells",
                  static_cast<int>(s.cells.size()));
    if (ImGui::TreeNodeEx(authoredLabel, ImGuiTreeNodeFlags_DefaultOpen)) {
        int remove = -1;
        for (size_t i = 0; i < s.cells.size(); ++i) {
            atmosphere::StormCell& c = s.cells[i];
            ImGui::PushID(static_cast<int>(i));
            float xz[2] = {c.position_m.x, c.position_m.z};
            if (ImGui::DragFloat2("Position X/Z (m)", xz, 10.0f, -1000000.0f, 1000000.0f, "%.0f")) {
                c.position_m.x = xz[0];
                c.position_m.z = xz[1];
                changed = true;
            }
            changed |= ImGui::DragFloat("Radius (m)", &c.radius_m, 10.0f, 200.0f, 30000.0f, "%.0f");
            changed |= ImGui::SliderFloat("Intensity", &c.intensity, 0.0f, 1.0f, "%.2f");
            changed |= ImGui::DragFloat("Birth (s)", &c.birth_s, 1.0f, -100000.0f, 100000.0f, "%.0f");
            changed |= ImGui::DragFloat("Lifetime (s, 0 = always mature)", &c.lifetime_s,
                                        10.0f, 0.0f, 14400.0f, "%.0f");
            if (ImGui::SmallButton("Remove")) remove = static_cast<int>(i);
            ImGui::Separator();
            ImGui::PopID();
        }
        if (remove >= 0) {
            s.cells.erase(s.cells.begin() + remove);
            changed = true;
        }
        if (s.cells.size() < static_cast<size_t>(atmosphere::kMaxManualStormCells) &&
            ImGui::Button("Add cell at domain centre")) {
            atmosphere::StormCell c;
            c.position_m = s.center_m;
            c.radius_m = s.cell_radius_m;
            s.cells.push_back(c);
            changed = true;
        }
        ImGui::TreePop();
    }
    ImGui::TreePop();
    return changed;
}

// Local weather (docs/dev/ATMOSPHERE_LOCAL_WINDS_FOG.md §2-4). The switches
// are cloud state; the readout is the published map (what is drawn).
static bool drawLocalWeatherControls(atmosphere::CloudState& clouds) {
    bool changed = false;
    if (!ImGui::TreeNodeEx("Local weather / fog", ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }
    atmosphere::CloudLocalWeather& lw = clouds.local_weather;
    changed |= ImGui::Checkbox("Storm births follow the terrain", &lw.terrain_suitability);
    ImGui::TextWrapped("Windward slopes, ridges and sun-facing slopes raise the odds of a new "
                       "cell; the lee lowers them. Without a terrain, births stay uniform.");
    changed |= ImGui::Checkbox("Local fog", &lw.fog_enabled);
    ImGui::BeginDisabled(!lw.fog_enabled);
    changed |= ImGui::DragFloat("Fog strength", &lw.fog_strength, 0.01f, 0.0f, 10.0f, "%.2f");
    changed |= ImGui::Checkbox("Author the fog (ignore the weather)", &lw.fog_manual);
    ImGui::BeginDisabled(!lw.fog_manual);
    changed |= ImGui::SliderFloat("Valley fog", &lw.valley_fog, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Valley fog lift", &lw.valley_fog_lift, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Ground mist", &lw.post_rain_mist, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::TextWrapped("Radiation fog forms on calm, humid, clear nights and pools in hollows; "
                       "it lifts and burns off after sunrise. Mist follows rain: inside rain "
                       "shafts and over ground that has just been rained on.");
    if (const auto map = atmosphere::localFogMapSnapshot()) {
        const atmosphere::LocalFogDrivers& d = map->drivers;
        if (lw.fog_manual) {
            ImGui::TextDisabled("Authored: the weather drivers below are not used");
        }
        ImGui::Text("Night %.2f  calm %.2f  RH %.2f  layer cover %.2f",
                    d.night, d.calm, d.humidity, d.layer_cover);
        ImGui::Text("Fog over %.0f%% of a %.1f km map (%s), peak %.4f 1/m",
                    map->fogged_fraction * 100.0f, map->size_m / 1000.0f,
                    map->terrain ? "terrain" : "flat", map->max_extinction);
    }
    ImGui::TreePop();
    return changed;
}

static bool drawLightningControls(atmosphere::CloudState& clouds) {
    bool changed = drawStormControls(clouds);
    changed |= drawLocalWeatherControls(clouds);
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
        changed |= ImGui::SliderFloat("Leader formation fraction", &lightning.leader_fraction,
                                      0.05f, 0.65f, "%.2f");
        ImGui::TextWrapped("Leader formation: %.0f ms before the bright stroke. "
                           "This time scale can slow formation for scene presentation.",
                           lightning.duration_s * lightning.leader_fraction * 1000.0f);
        changed |= ImGui::DragFloat("Horizontal flash span (m)", &lightning.horizontal_span_m,
                                    100.0f, 100.0f, 50000.0f, "%.0f");
        changed |= ImGui::SliderFloat("Branching", &lightning.branching, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::DragFloat("Channel brightness", &lightning.channel_brightness,
                                    0.1f, 0.0f, 1000.0f, "%.1f");
        const float effectiveRate = Backend::lightningEffectiveRate(lightning);
        if (effectiveRate + 0.1f < lightning.rate_per_minute) {
            ImGui::TextWrapped("At this duration: up to %.0f flashes/min. "
                               "Shorter duration allows more frequent flashes.", effectiveRate);
        }
        changed |= ImGui::DragFloat("Flash intensity", &lightning.intensity,
                                    0.1f, 0.0f, 10000.0f, "%.1f");
        changed |= ImGui::ColorEdit3("Flash colour", &lightning.color.x,
                                     ImGuiColorEditFlags_Float);
        changed |= ImGui::Checkbox("Strikes attach to the scene", &lightning.attach_to_scene);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Ground strikes hit terrain, meshes and scattered trees: of what lies "
                              "within the striking distance, the point a descending sphere touches "
                              "first (rolling sphere).");
        changed |= ImGui::DragFloat("Fallback strike altitude (m)", &lightning.ground_altitude_m,
                                    10.0f, -500.0f, 20000.0f, "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Where a strike ends when nothing in the scene lies under it "
                              "(or attachment is off).");
        changed |= ImGui::SliderFloat("Ground strike fraction", &lightning.ground_strike_fraction,
                                      0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Upward share of ground strikes", &lightning.upward_fraction,
                                      0.0f, 1.0f, "%.2f");
        ImGui::TextWrapped("Lower ground strike fraction produces more wide, horizontal "
                           "cloud-to-cloud flashes. Longer flashes develop their branches over time. "
                           "Downward strikes branch toward the ground; upward ones (from peaks and "
                           "towers) start on the ground and branch up into the cloud.");
        changed |= ImGui::DragFloat("Bolt radius (m)", &lightning.bolt_radius_m,
                                    0.1f, 0.1f, 30.0f, "%.1f");
        ImGui::TextWrapped("Vulkan RT and RayFusion share seeded timeline flashes. "
                           "Flashes come only from MATURE storm cells (Storms below): "
                           "no cells, no lightning. Use timeline Play to see flashes. "
                           "Realtime Weather Preview animates rain/snow, not lightning.");
        ImGui::TreePop();
    }
    return changed;
}

bool drawMaterialControls(atmosphere::CloudState& clouds) {
    return drawOpticalControls(clouds.material);
}

bool drawWeatherLightningControls(UIContext& ctx) {
    atmosphere::CloudState clouds = ctx.renderer.world.getClouds();
    bool changed = drawLightningControls(clouds);
    static std::string error;
    if (freezeRequested) {
        freezeRequested = false;
        // Same action as IPC world.set_clouds {storms:{freeze_automatic:true}}.
        const rtapi::Result result = rtapi::setCloudsJson(R"({"storms":{"freeze_automatic":true}})");
        error = result.error;
        return result.ok;
    }
    if (changed) {
        nlohmann::json json;
        atmosphere::cloudsToJson(clouds, json);
        // Every block drawLightningControls edits must be in the patch: a
        // block left out snaps back next frame (local_weather did, then
        // volumes did -- drawStormControls edits clouds.volumes too).
        const rtapi::Result result = rtapi::setCloudsJson(
            nlohmann::json{{"lightning", json["lightning"]}, {"storms", json["storms"]},
                           {"local_weather", json["local_weather"]},
                           {"volumes", json["volumes"]}}.dump());
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
        changed |= ImGui::SliderInt("Realtime interleave", &clouds.quality.realtime_interleave,
                                   1, 4);
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
