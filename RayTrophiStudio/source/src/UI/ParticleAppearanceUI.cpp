#include "UI/ParticleAppearanceUI.h"

#include "Api/RtApi.h"
#include "ParticleSimulation.h"
#include "scene_ui.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace ParticleAppearanceUI {
namespace {

// Last rtapi error, shown under the editor until the next successful write.
std::string g_error;

rtapi::ParticleSystemRef refFor(uint32_t system_id) {
    rtapi::ParticleSystemRef ref;
    ref.id = static_cast<int>(system_id);
    return ref;
}

std::string emitterRef(const RayTrophiSim::ParticleEmitterDesc& emitter) {
    return "uid:" + std::to_string(emitter.timeline_uid);
}

void note(const rtapi::Result& r) {
    g_error = r.ok ? std::string() : r.error;
}

void assignProfile(uint32_t system_id, const RayTrophiSim::ParticleEmitterDesc& emitter,
                   uint32_t profile_id) {
    const rtapi::ParticleSystemRef ref = refFor(system_id);
    rtapi::ParticleEmitterInfo info;
    rtapi::Result r = rtapi::getParticleEmitter(emitterRef(emitter), info, ref);
    if (r.ok) {
        info.appearance_profile_id = profile_id;
        r = rtapi::updateParticleEmitter(emitterRef(emitter), info, ref);
    }
    note(r);
}

const float* lutRowFor(UIContext& ctx, uint32_t system_id, uint32_t profile_id) {
    for (const auto& system : ctx.scene.particle_systems) {
        if (system.id == system_id && system.runtime) {
            return system.runtime->appearanceLutRow(profile_id);
        }
    }
    return nullptr;
}

// Colour strip (rgb * emission) over a checker, then an opacity strip.
void drawPreview(const float* row) {
    if (!row) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(64.0f, ImGui::GetContentRegionAvail().x);
    const float height = 14.0f;
    const int segments = RayTrophiSim::kParticleAppearanceLutSamples;
    const float step = width / static_cast<float>(segments);
    auto channel = [](float v) {
        return static_cast<int>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    for (int i = 0; i < segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments - 1);
        const RayTrophiSim::ParticleAppearanceSample s =
            RayTrophiSim::sampleParticleAppearanceLut(row, t);
        const ImVec2 a(origin.x + step * i, origin.y);
        const ImVec2 b(origin.x + step * (i + 1), origin.y + height);
        const bool dark = ((i / 2) % 2) == 0;
        dl->AddRectFilled(a, b, dark ? IM_COL32(60, 60, 60, 255) : IM_COL32(110, 110, 110, 255));
        const Vec3 c = s.color * s.emission;
        dl->AddRectFilled(a, b, IM_COL32(channel(c.x), channel(c.y), channel(c.z),
                                         channel(s.opacity)));
        const ImVec2 oa(a.x, origin.y + height + 2.0f);
        const ImVec2 ob(b.x, origin.y + height * 2.0f + 2.0f);
        const int g = channel(s.opacity);
        dl->AddRectFilled(oa, ob, IM_COL32(g, g, g, 255));
    }
    ImGui::Dummy(ImVec2(width, height * 2.0f + 4.0f));
    const RayTrophiSim::ParticleAppearanceSample birth =
        RayTrophiSim::sampleParticleAppearanceLut(row, 0.0f);
    const RayTrophiSim::ParticleAppearanceSample death =
        RayTrophiSim::sampleParticleAppearanceLut(row, 1.0f);
    ImGui::TextDisabled("colour x emission / opacity over life   size %.3f -> %.3f m",
                        birth.size, death.size);
}

// Returns true when a key changed. `lo`/`hi` bound the value.
bool editCurve(const char* label, std::vector<rtapi::ParticleCurveKeyInfo>& keys,
               float lo, float hi, float speed) {
    bool changed = false;
    if (!ImGui::TreeNode(label)) return false;
    int remove = -1;
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
        ImGui::PushID(i);
        float pair[2] = {keys[i].t, keys[i].value};
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 28.0f);
        if (ImGui::DragFloat2("##key", pair, speed, lo, hi, "%.3f")) {
            keys[i].t = std::clamp(pair[0], 0.0f, 1.0f);
            keys[i].value = std::clamp(pair[1], lo, hi);
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("age (0 = birth, 1 = death), value");
        ImGui::SameLine();
        if (ImGui::SmallButton("x") && keys.size() > 1) remove = i;
        ImGui::PopID();
    }
    if (remove >= 0) {
        keys.erase(keys.begin() + remove);
        changed = true;
    }
    if (keys.size() < RayTrophiSim::kParticleAppearanceMaxKeys && ImGui::SmallButton("+ key")) {
        const rtapi::ParticleCurveKeyInfo last =
            keys.empty() ? rtapi::ParticleCurveKeyInfo{0.0f, lo} : keys.back();
        keys.push_back({std::min(1.0f, last.t + 0.25f), last.value});
        changed = true;
    }
    ImGui::TreePop();
    return changed;
}

bool editRamp(std::vector<rtapi::ParticleColorStopInfo>& stops) {
    bool changed = false;
    if (!ImGui::TreeNode("Colour Ramp")) return false;
    int remove = -1;
    for (int i = 0; i < static_cast<int>(stops.size()); ++i) {
        ImGui::PushID(i);
        ImGui::SetNextItemWidth(70.0f);
        if (ImGui::DragFloat("##t", &stops[i].t, 0.005f, 0.0f, 1.0f, "%.3f")) {
            stops[i].t = std::clamp(stops[i].t, 0.0f, 1.0f);
            changed = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 28.0f);
        if (ImGui::ColorEdit3("##c", &stops[i].color.x, ImGuiColorEditFlags_HDR |
                                                          ImGuiColorEditFlags_Float)) {
            stops[i].color = Vec3(std::max(0.0f, stops[i].color.x),
                                  std::max(0.0f, stops[i].color.y),
                                  std::max(0.0f, stops[i].color.z));
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("x") && stops.size() > 1) remove = i;
        ImGui::PopID();
    }
    if (remove >= 0) {
        stops.erase(stops.begin() + remove);
        changed = true;
    }
    if (stops.size() < RayTrophiSim::kParticleAppearanceMaxKeys && ImGui::SmallButton("+ stop")) {
        const rtapi::ParticleColorStopInfo last =
            stops.empty() ? rtapi::ParticleColorStopInfo{} : stops.back();
        stops.push_back({std::min(1.0f, last.t + 0.25f), last.color});
        changed = true;
    }
    ImGui::TreePop();
    return changed;
}

void drawManageProfiles(uint32_t system_id,
                        const std::vector<rtapi::ParticleAppearanceInfo>& profiles) {
    if (!ImGui::TreeNode("Profiles in this system")) return;
    for (const auto& p : profiles) {
        ImGui::PushID(static_cast<int>(p.id));
        ImGui::Text("#%u  %s  (%s, %d emitter%s)", p.id, p.name.c_str(), p.blend.c_str(),
                    static_cast<int>(p.used_by_emitter_uids.size()),
                    p.used_by_emitter_uids.size() == 1 ? "" : "s");
        if (p.used_by_emitter_uids.empty()) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove")) {
                note(rtapi::removeParticleAppearance(p.id, refFor(system_id)));
            }
        }
        ImGui::PopID();
    }
    ImGui::TreePop();
}

} // namespace

void drawEmitterAppearance(UIContext& ctx, uint32_t system_id,
                           const RayTrophiSim::ParticleEmitterDesc& emitter) {
    const rtapi::ParticleSystemRef ref = refFor(system_id);
    std::vector<rtapi::ParticleAppearanceInfo> profiles;
    if (rtapi::Result r = rtapi::listParticleAppearances(ref, profiles); !r.ok) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "%s", r.error.c_str());
        return;
    }

    const rtapi::ParticleAppearanceInfo* current = nullptr;
    for (const auto& p : profiles) {
        if (p.id == emitter.appearance_profile_id) current = &p;
    }

    // ── Profile picker ─────────────────────────────────────────────────────
    char label[160];
    std::snprintf(label, sizeof(label), "%s (#%u)",
                  current ? current->name.c_str() : "<missing>",
                  emitter.appearance_profile_id);
    if (ImGui::BeginCombo("Appearance##PartAppearance", label)) {
        for (const auto& p : profiles) {
            char item[160];
            std::snprintf(item, sizeof(item), "%s (#%u)##prof%u", p.name.c_str(), p.id, p.id);
            if (ImGui::Selectable(item, p.id == emitter.appearance_profile_id) &&
                p.id != emitter.appearance_profile_id) {
                assignProfile(system_id, emitter, p.id);
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("The look over life. Several emitters may share one profile;\n"
                          "editing it changes all of them.");
    }
    if (current) {
        if (ImGui::SmallButton("Duplicate##PartAppDup")) {
            rtapi::ParticleAppearanceInfo copy = *current;
            copy.name = current->name + " Copy";
            rtapi::ParticleAppearanceInfo created;
            rtapi::Result r = rtapi::addParticleAppearance(copy, created, ref);
            note(r);
            if (r.ok) assignProfile(system_id, emitter, created.id);
            return;  // `profiles` is stale now
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Give this emitter its own copy, leaving other emitters untouched.");
        }
        ImGui::SameLine();
        ImGui::TextDisabled("shared by %d emitter(s)",
                            static_cast<int>(current->used_by_emitter_uids.size()));
    }
    drawManageProfiles(system_id, profiles);
    if (!current) {
        if (!g_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "%s", g_error.c_str());
        }
        return;
    }

    // ── Editor ─────────────────────────────────────────────────────────────
    rtapi::ParticleAppearanceInfo edited = *current;
    bool changed = false;

    char name[128];
    std::snprintf(name, sizeof(name), "%s", edited.name.c_str());
    if (ImGui::InputText("Profile Name##PartAppName", name, sizeof(name),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        edited.name = name;
        changed = true;
    }
    int blend = edited.blend == "alpha" ? 1 : 0;
    const char* blend_names[] = {"Additive (fire / spark glow)", "Alpha (smoke / dust)"};
    if (ImGui::Combo("Blend##PartAppBlend", &blend, blend_names, IM_ARRAYSIZE(blend_names))) {
        edited.blend = blend == 1 ? "alpha" : "additive";
        changed = true;
    }

    drawPreview(lutRowFor(ctx, system_id, current->id));
    changed |= editRamp(edited.color_ramp);
    changed |= editCurve("Opacity##PartAppOpacity", edited.opacity_curve, 0.0f, 1.0f, 0.005f);
    changed |= editCurve("Size (m)##PartAppSize", edited.size_curve, 0.0f, 1000.0f, 0.002f);
    changed |= editCurve("Emission##PartAppEmission", edited.emission_curve, 0.0f, 1000.0f, 0.02f);

    if (changed) {
        note(rtapi::updateParticleAppearance(current->id, edited, ref));
    }
    if (!g_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "%s", g_error.c_str());
    }
}

} // namespace ParticleAppearanceUI
