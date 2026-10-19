#include "UI/AudioHistoryUI.h"
#include "UI/AudioSoundDesignerUI.h"
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <array>
#include <cstdint>
#include <cstring>

namespace AudioUI {
namespace {
const audio::AudioService* owner = nullptr;
std::uint64_t generation = 0;
std::array<ImGuiID, 2> windows{};

void syncOwner(UIContext& ctx) {
    if (owner != ctx.scene.audio.get() || generation != ctx.scene.audio->generation()) {
        owner = ctx.scene.audio.get();
        generation = ctx.scene.audio->generation();
        windows = {};
    }
}

bool designerTarget(UIContext& ctx) {
    if (!ctx.scene.audio || !ImGui::GetCurrentContext()) {
        return false;
    }
    syncOwner(ctx);
    auto workspace = ctx.scene.audio->invoke("workspace_get", nlohmann::json::object());
    const auto page = workspace.value("designer_page", "generate");
    const bool live = page == "analyze" && workspace.value("analysis_mode", "live") == "live";
    const bool arrange = page == "edit" &&
        workspace.value("designer_edit_view", "source") == "arrange";
    bool selected = workspace.value("history_scope", "scene") == "design";
    auto* focused = ImGui::GetCurrentContext()->NavWindow;
    // Menus temporarily take focus; retain the owner which opened the menu.
    const bool menu = focused && ((focused->Flags & ImGuiWindowFlags_Popup) ||
        std::strcmp(focused->Name, "##MainMenuBar") == 0);
    if (!menu) {
        selected = false;
        for (auto* window = focused; window; window = window->ParentWindow) {
            for (auto id : windows) {
                selected = selected || (id != 0 && window->ID == id);
            }
        }
    }
    if (!workspace.value("designer_open", false) || live || arrange) {
        selected = false;
    }
    const char* scope = selected ? "design" : "scene";
    if (workspace.value("history_scope", "scene") != scope) {
        workspaceCall(ctx, "workspace_update", {{"history_scope", scope}});
    }
    return selected;
}
}

void registerAudioHistoryWindow(UIContext& ctx, bool inspector) {
    syncOwner(ctx);
    windows[inspector ? 1 : 0] = ImGui::GetCurrentWindow()->ID;
    designerTarget(ctx);
}

bool audioHistoryOwnsFocus(UIContext& ctx) {
    return designerTarget(ctx);
}

bool drawAudioHistoryMenu(UIContext& ctx, bool blocked) {
    if (!designerTarget(ctx)) {
        return false;
    }
    const bool typing = ImGui::GetIO().WantTextInput;
    if (ImGui::MenuItem("Undo Sound Designer", "Ctrl+Z", false,
            !blocked && !typing && canSoundDesignerHistory(ctx, false))) {
        stepSoundDesignerHistory(ctx, false);
    }
    if (ImGui::MenuItem("Redo Sound Designer", "Ctrl+Y", false,
            !blocked && !typing && canSoundDesignerHistory(ctx, true))) {
        stepSoundDesignerHistory(ctx, true);
    }
    return true;
}

bool handleAudioHistoryShortcuts(UIContext& ctx, bool blocked) {
    if (ImGui::GetIO().WantTextInput) {
        return true;
    }
    if (!designerTarget(ctx)) {
        return false;
    }
    const auto& io = ImGui::GetIO();
    const bool undo = io.KeyCtrl && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z);
    const bool redo = io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_Y) ||
        (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z)));
    if (!blocked && (undo || redo) && canSoundDesignerHistory(ctx, redo)) {
        stepSoundDesignerHistory(ctx, redo);
    }
    return true; // Empty Designer history must not fall through to scene undo.
}
}
