#include "Audio/AcousticRenderDebug.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"

#include <exception>
#include <string>

namespace audio {
void drawAcousticRenderDebugControls(UIContext& ctx) {
    if (ctx.render_settings.debug_view != kAcousticSurfaceDebugView) {
        return;
    }
    const auto data = ctx.scene.audio->snapshot();
    const auto selected = ctx.selection.selected.type == SelectableType::AudioSource ?
        ctx.selection.selected.audio_source_id : std::string();
    std::string label = "Select a spatial manual source";
    for (const auto& emitter : data.at("emitters")) {
        if (emitter.at("id") == selected) {
            label = emitter.value("name", selected);
        }
    }
    static std::string error;
    if (ImGui::BeginCombo("Acoustic source", label.c_str())) {
        for (const auto& emitter : data.at("emitters")) {
            if (!emitter.value("spatial", false) ||
                emitter.at("reaction").value("driver", "manual") != "manual") {
                continue;
            }
            const auto id = emitter.at("id").get<std::string>();
            const auto name = emitter.value("name", id) + "##" + id;
            if (ImGui::Selectable(name.c_str(), id == selected)) {
                try {
                    author(ctx, nullptr, "select", {{"id", id}});
                    ctx.start_render = true;
                    error.clear();
                } catch (const std::exception& exception) {
                    error = exception.what();
                }
            }
        }
        ImGui::EndCombo();
    }
    ImGui::TextWrapped("Surface field: purple -60 dB, yellow 0 dB relative to unit gain. "
        "Black: blocked or zero energy. Grey: no eligible source. Exposure scales energy.");
    ImGui::TextWrapped("Authored source level, including distance and cone; playback is optional. "
        "Active-AS authored solids are blockers. No wall transmission, reflections, "
        "diffraction, frequency bands or calibrated SPL. Vulkan RT only.");
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
}
} // namespace audio
