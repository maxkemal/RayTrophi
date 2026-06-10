// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - VIEWPORT OVERLAYS
// ═══════════════════════════════════════════════════════════════════════════════
// This file contains viewport overlay components:
//   - drawFocusIndicator()  : Split-prism focus aid with focus ring
//   - drawZoomRing()        : FOV control ring
//   - drawDollyArc()        : Camera dolly track control (disabled)
//   - drawExposureInfo()    : Exposure triangle with AE toggle
//   - drawViewportControls(): viewport overlay buttons
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "scene_data.h"   // Explicit include
#include "Triangle.h"
#include "InstanceManager.h"
#include "imgui.h"
#include "CameraPresets.h"
#include "Backend/IViewportBackend.h"
#include "Backend/OptixBackend.h"
#include "Backend/VulkanBackend.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "ProjectManager.h"

extern bool g_hasVulkan;
extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

// extern ProjectManager g_ProjectManager; - Removed to use Singleton access



// ═════════════════════════════════════════════════════════════════════════════
// VIEWPORT CONTROLS OVERLAY  (compact top-center header bar)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawViewportControls(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    SceneSelection& sel = ctx.selection;

    // ── Layout constants ──
    const float menu_height = 19.0f;
    const float btn_h = 24.0f;
    const float pad_x = 6.0f;

    const float right_margin = 18.0f + getPaintBrushDockWidth();
    const float top_margin = menu_height + 10.0f;

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - right_margin, top_margin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(0, 0)); // auto-size

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.09f, 0.11f, 0.14f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.72f, 0.78f, 0.88f, 0.14f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(pad_x, 3.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (!ImGui::Begin("##ViewportBar", nullptr, flags)) {
        ImGui::PopStyleVar(4);
        ImGui::PopStyleColor(2);
        ImGui::End();
        return;
    }

    auto iconToggleBtn = [&](const char* id, bool active, ImVec2 size, ImVec4 accent, auto&& drawIcon) -> bool {
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const bool hovered_pre = ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        const float mix = active ? 0.78f : (hovered_pre ? 0.42f : 0.20f);
        const ImVec4 base(0.18f, 0.19f, 0.22f, 0.90f);
        const ImVec4 bg(
            base.x + (accent.x - base.x) * mix,
            base.y + (accent.y - base.y) * mix,
            base.z + (accent.z - base.z) * mix,
            active ? 0.98f : 0.90f);

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(
            std::min(1.0f, bg.x + 0.10f),
            std::min(1.0f, bg.y + 0.10f),
            std::min(1.0f, bg.z + 0.10f),
            1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(
            std::min(1.0f, bg.x + 0.05f),
            std::min(1.0f, bg.y + 0.05f),
            std::min(1.0f, bg.z + 0.05f),
            1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, active ? 0.75f : 0.25f));

        const bool clicked = ImGui::Button(id, size);
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawIcon(dl, min, max, ImGui::ColorConvertFloat4ToU32(ImVec4(0.92f, 0.94f, 0.98f, active ? 1.0f : 0.92f)));
        dl->AddRect(min, max, ImGui::ColorConvertFloat4ToU32(ImVec4(accent.x, accent.y, accent.z, active ? 0.45f : 0.18f)), 8.0f, 0, 1.0f);

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
        return clicked;
    };
    auto iconButton = [&](const char* id, ImVec2 size, ImVec4 accent, auto&& drawIcon) -> bool {
        return iconToggleBtn(id, false, size, accent, drawIcon);
    };
    auto iconToggleTypeBtn = [&](const char* id, bool active, ImVec2 size, ImVec4 accent, UIWidgets::IconType type, float iconSize = 22.0f) -> bool {
        return iconToggleBtn(id, active, size, accent,
            [&](ImDrawList*, ImVec2 min, ImVec2 max, ImU32 col) {
                const ImVec2 pos((min.x + max.x - iconSize) * 0.5f, (min.y + max.y - iconSize) * 0.5f);
                UIWidgets::DrawIcon(type, pos, iconSize, col, 1.45f);
            });
    };
    auto iconTypeButton = [&](const char* id, ImVec2 size, ImVec4 accent, UIWidgets::IconType type, float iconSize = 22.0f) -> bool {
        return iconToggleTypeBtn(id, false, size, accent, type, iconSize);
    };
    auto viewportTooltip = [&](const char* title, const char* body, const ImVec4& accent) {
        if (!ImGui::IsItemHovered()) return;
        ImGui::SetNextWindowBgAlpha(0.78f);
        ImGui::SetNextWindowSizeConstraints(ImVec2(260.0f, 0.0f), ImVec2(360.0f, 1000.0f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.07f, 0.08f, 0.10f, 0.82f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, 0.22f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, accent);
        ImGui::TextUnformatted(title);
        ImGui::PopStyleColor();
        if (body && body[0]) {
            ImGui::Separator();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextWrapped("%s", body);
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTooltip();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    };

    // ── Transform gizmo mode buttons ──
    bool is_t = (sel.transform_mode == TransformMode::Translate);
    bool is_r = (sel.transform_mode == TransformMode::Rotate);
    bool is_s = (sel.transform_mode == TransformMode::Scale);
    const ImVec4 tCol(0.3f, 0.7f, 1.0f, 1.0f);

    if (iconToggleTypeBtn("##transform_translate", is_t, ImVec2(btn_h + 8.0f, btn_h), tCol, UIWidgets::IconType::Move, 22.0f)) sel.transform_mode = TransformMode::Translate;
    viewportTooltip("Move", "Translate the current selection.", tCol);
    ImGui::SameLine();
    if (iconToggleTypeBtn("##transform_rotate", is_r, ImVec2(btn_h + 8.0f, btn_h), tCol, UIWidgets::IconType::Rotate, 22.0f)) sel.transform_mode = TransformMode::Rotate;
    viewportTooltip("Rotate", "Rotate the current selection.", tCol);
    ImGui::SameLine();
    if (iconToggleTypeBtn("##transform_scale", is_s, ImVec2(btn_h + 8.0f, btn_h), tCol, UIWidgets::IconType::ScaleAxis, 22.0f)) sel.transform_mode = TransformMode::Scale;
    viewportTooltip("Scale", "Scale the current selection.", tCol);

    // Gizmo toggle
    ImGui::SameLine();
    if (iconToggleTypeBtn("##toggle_gizmo", viewport_settings.show_gizmos, ImVec2(btn_h + 8.0f, btn_h), ImVec4(0.57f, 0.71f, 1.0f, 1.0f), UIWidgets::IconType::Gizmo, 22.0f))
    {
        viewport_settings.show_gizmos = !viewport_settings.show_gizmos;
        ProjectManager::getInstance().markModified();
    }
    viewportTooltip("Gizmos", "Show or hide transform gizmos in the viewport.\nRight-click for outline options.", ImVec4(0.57f, 0.71f, 1.0f, 1.0f));
    if (ImGui::BeginPopupContextItem("##gizmo_opts")) {
        if (ImGui::MenuItem("Selection Outline", nullptr, viewport_settings.show_selection_outline)) {
            viewport_settings.show_selection_outline = !viewport_settings.show_selection_outline;
            ProjectManager::getInstance().markModified();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Per-mesh silhouette around the selected object.\nDisable for a large viewport perf win on dense selections.");
        }
        ImGui::EndPopup();
    }

    // ── Separator ──
    ImGui::SameLine(0, 6.0f);
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
    ImGui::SameLine(0, 6.0f);

    // ── Viewport Shading Buttons ──
    {
        struct ShadingBtn {
            int mode;
            const char* id;
            const char* title;
            const char* tooltip;
            ImVec4 accent;
            bool enabled;
        };
        const bool hasRasterViewport = (g_viewport_backend != nullptr) ||
            (ctx.backend_ptr && ctx.backend_ptr->supportsViewportMode(Backend::ViewportMode::Solid));
        const ShadingBtn btns[] = {
            { 0, "##shade_solid",   "Solid",   hasRasterViewport ? "Fast raster preview for layout work." : "Requires Vulkan — not available on this machine.",              ImVec4(0.94f, 0.64f, 0.24f, 1.0f), hasRasterViewport  },
            { 3, "##shade_matcap",  "Matcap",  hasRasterViewport ? "Studio-style shaded preview with matcap lighting." : "Requires Vulkan — not available on this machine.", ImVec4(0.88f, 0.42f, 0.34f, 1.0f), hasRasterViewport  },
            { 1, "##shade_preview", "Preview", hasRasterViewport ? "PBR material preview with stable studio/environment lighting." : "Requires Vulkan — not available on this machine.", ImVec4(0.34f, 0.72f, 0.62f, 1.0f), hasRasterViewport },
            { 2, "##shade_render",  "Render",  "Full rendered viewport using the selected device.",  ImVec4(0.32f, 0.60f, 0.96f, 1.0f), true  },
        };
        static float hover_anim[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        const float dt = ImGui::GetIO().DeltaTime;
        const float shade_btn_w = 50.0f;
        const float shade_btn_h = btn_h + 4.0f;
        const float shade_icon_base = 28.0f;
        const float shade_icon_hover_boost = 2.5f;

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 0.0f));
        for (int i = 0; i < 4; ++i) {
            if (i > 0) ImGui::SameLine();
            const bool is_active = (viewport_settings.shading_mode == btns[i].mode);
            const ImVec4 accent = btns[i].accent;
            const bool is_enabled = btns[i].enabled;

            const ImVec2 shade_btn_pos = ImGui::GetCursorScreenPos();
            const bool hovered_pre = ImGui::IsMouseHoveringRect(
                shade_btn_pos,
                ImVec2(shade_btn_pos.x + shade_btn_w, shade_btn_pos.y + shade_btn_h));
            const float hover_target = ((hovered_pre && is_enabled) || is_active) ? 1.0f : 0.0f;
            hover_anim[i] += (hover_target - hover_anim[i]) * std::min(1.0f, dt * 12.0f);

            bool clicked = false;
            if (!is_enabled) ImGui::BeginDisabled();
            UIWidgets::IconType icon_type = UIWidgets::IconType::ViewRendered;
            if (btns[i].mode == 0) icon_type = UIWidgets::IconType::ViewSolid;
            else if (btns[i].mode == 3) icon_type = UIWidgets::IconType::ViewMatcap;
            else if (btns[i].mode == 1) icon_type = UIWidgets::IconType::ViewPreview;
            else if (btns[i].mode == 2) icon_type = UIWidgets::IconType::ViewRendered;
            clicked = iconToggleTypeBtn(
                btns[i].id,
                is_active,
                ImVec2(shade_btn_w, shade_btn_h),
                is_enabled ? accent : ImVec4(0.34f, 0.38f, 0.42f, 1.0f),
                icon_type,
                shade_icon_base + hover_anim[i] * shade_icon_hover_boost);
            if (!is_enabled) ImGui::EndDisabled();

            if (clicked && is_enabled) {
                Backend::ViewportMode requestedMode = Backend::ViewportMode::Rendered;
                switch (btns[i].mode) {
                    case 0: requestedMode = Backend::ViewportMode::Solid; break;
                    case 1: requestedMode = Backend::ViewportMode::MaterialPreview; break;
                    case 3: requestedMode = Backend::ViewportMode::Matcap; break;
                    case 2:
                    default: requestedMode = Backend::ViewportMode::Rendered; break;
                }
                const bool supported =
                    (btns[i].mode == 2) ||
                    (g_viewport_backend != nullptr) ||
                    (ctx.backend_ptr && ctx.backend_ptr->supportsViewportMode(requestedMode));

                if (!supported) {
                    viewport_settings.shading_mode = 2;
                    addViewportMessage("Interactive viewport is not available right now. Switched to Rendered mode.",
                        3.0f, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
                } else {
                    viewport_settings.shading_mode = btns[i].mode;
                    if (viewport_settings.shading_mode != 2 && g_viewport_backend != nullptr) {
                        g_viewport_raster_rebuild_pending = true;
                    }
                }
                ctx.start_render = true;
                ctx.renderer.resetCPUAccumulation();
                if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
            }
            const bool hovered = ImGui::IsItemHovered();

            if (hovered) {
                std::string tooltip_text = btns[i].tooltip;
                if (!is_enabled) {
                    tooltip_text += " Coming soon.";
                } else if (btns[i].mode == 0 || btns[i].mode == 3) {
                    tooltip_text += " Early viewport mode.";
                }
                viewportTooltip(btns[i].title, tooltip_text.c_str(), accent);
            }
        }
        ImGui::PopStyleVar();

        // Matcap gear popup
        if (viewport_settings.shading_mode == 3) {
            ImGui::SameLine(0, 3.0f);
            if (iconTypeButton("##mcap_preset", ImVec2(btn_h + 4.0f, btn_h), ImVec4(0.94f, 0.58f, 0.38f, 1.0f), UIWidgets::IconType::Settings, 18.0f))
                ImGui::OpenPopup("MatcapPresetPopup");
            viewportTooltip("Matcap Preset", "Choose the active matcap surface preset.", ImVec4(0.94f, 0.58f, 0.38f, 1.0f));
            if (ImGui::BeginPopup("MatcapPresetPopup")) {
                static const int pv[] = { 0, 2, 3, 4, 5, 6, 7, 8, 9 };
                static const char* pl[] = { "Solid Clay","Default","Clay","Silver","Pearl","Jade","Copper","Obsidian","Skin" };
                for (int i = 0; i < IM_ARRAYSIZE(pv); ++i) {
                    bool s = (viewport_settings.matcap_preset == pv[i]);
                    if (ImGui::Selectable(pl[i], s)) {
                        viewport_settings.matcap_preset = pv[i];
                        Backend::IBackend* matcapBackend = ctx.backend_ptr;
                        if (viewport_settings.shading_mode != 2 &&
                            g_viewport_backend &&
                            g_viewport_backend.get() != ctx.backend_ptr) {
                            matcapBackend = g_viewport_backend.get();
                        }
                        if (matcapBackend) {
                            matcapBackend->setInteractiveViewportMatcapPreset(pv[i]);
                        }
                        ctx.start_render = true;
                        ctx.renderer.resetCPUAccumulation();
                    }
                    if (s) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndPopup();
            }
        }
    }

    // ── Separator ──
    ImGui::SameLine(0, 6.0f);
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
    ImGui::SameLine(0, 6.0f);

    // ── HUD / PRO ──
    const ImVec4 hudCol(0.2f, 0.6f, 0.4f, 1.0f);
    if (iconToggleTypeBtn("##toggle_hud", viewport_settings.show_camera_hud, ImVec2(btn_h + 8.0f, btn_h), hudCol, UIWidgets::IconType::CameraHud, 22.0f)) {
        viewport_settings.show_camera_hud = !viewport_settings.show_camera_hud;
        ProjectManager::getInstance().markModified();
    }
    viewportTooltip("Camera HUD", "Show lens and framing info in the viewport.", hudCol);

    ImGui::SameLine();
    bool any_pro = viewport_settings.show_histogram || viewport_settings.show_focus_peaking ||
                   viewport_settings.show_zebra || viewport_settings.show_af_points;
    if (iconToggleTypeBtn("##toggle_pro", any_pro, ImVec2(btn_h + 8.0f, btn_h), ImVec4(0.6f, 0.3f, 0.5f, 1.0f), UIWidgets::IconType::ViewOverlays, 22.0f))
        ImGui::OpenPopup("ProCameraPopup");
    viewportTooltip("Overlays", "Histogram, focus peaking, zebra and AF tools.", ImVec4(0.6f, 0.3f, 0.5f, 1.0f));

    if (ImGui::BeginPopup("ProCameraPopup")) {
        ImGui::Text("Pro Camera Features");
        ImGui::Separator();
        ImGui::Checkbox("Histogram", &viewport_settings.show_histogram);
        if (viewport_settings.show_histogram) {
            ImGui::Indent();
            ImGui::Combo("Mode##Hist", &viewport_settings.histogram_mode, "RGB\0Luma\0");
            ImGui::SliderFloat("Opacity", &viewport_settings.histogram_opacity, 0.3f, 1.0f);
            ImGui::Unindent();
        }
        ImGui::Separator();
        ImGui::Checkbox("Focus Peaking", &viewport_settings.show_focus_peaking);
        if (viewport_settings.show_focus_peaking) {
            ImGui::Indent();
            ImGui::Combo("Color##Peak", &viewport_settings.focus_peaking_color, "Red\0Yellow\0Green\0Blue\0White\0");
            ImGui::SliderFloat("Threshold##Peak", &viewport_settings.focus_peaking_threshold, 0.05f, 0.5f);
            ImGui::Unindent();
        }
        ImGui::Separator();
        ImGui::Checkbox("Zebra Stripes", &viewport_settings.show_zebra);
        if (viewport_settings.show_zebra) {
            ImGui::Indent();
            ImGui::SliderFloat("Threshold##Zebra", &viewport_settings.zebra_threshold, 0.8f, 1.0f, "%.2f");
            ImGui::Unindent();
        }
        ImGui::Separator();
        ImGui::Checkbox("AF Points", &viewport_settings.show_af_points);
        if (viewport_settings.show_af_points) {
            ImGui::Indent();
            ImGui::Combo("Mode##AF", &viewport_settings.af_mode, "Single\0Zone 9\0Zone 21\0Wide\0Center Weighted\0");
            ImGui::Combo("Focus Mode", &viewport_settings.focus_mode, "MF (Manual)\0AF-S (Single)\0AF-C (Continuous)\0");
            ImGui::Unindent();
        }
        ImGui::EndPopup();
    }

    // ── Separator ──
    ImGui::SameLine(0, 6.0f);
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
    ImGui::SameLine(0, 6.0f);

    // ── Pivot ──
    {
        const bool can_edit_object_pivot =
            sel.selected.type == SelectableType::Object &&
            sel.selected.object && sel.multi_selection.size() == 1;
        const bool can_edit_volume_pivot =
            sel.multi_selection.size() == 1 &&
            ((sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) ||
             (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume));
        const bool can_edit_any_pivot = can_edit_object_pivot || can_edit_volume_pivot;

        ImGui::SetNextItemWidth(76);
        const char* pivot_opts[] = { "Median", "Individual" };
        ImGui::Combo("##Pivot", &pivot_mode, pivot_opts, 2);
        viewportTooltip("Pivot Mode", "Choose how transforms use pivots.", ImVec4(0.78f, 0.68f, 0.32f, 1.0f));

        ImGui::SameLine();
        if (iconToggleTypeBtn("##toggle_pivot_edit", pivot_edit_mode, ImVec2(btn_h, btn_h), ImVec4(0.85f, 0.55f, 0.18f, 1.0f), UIWidgets::IconType::PivotEdit, 20.0f)) {
            pivot_edit_mode = can_edit_any_pivot ? !pivot_edit_mode : false;
        }
        viewportTooltip("Edit Pivot", "Move the pivot without moving the object.", ImVec4(0.85f, 0.55f, 0.18f, 1.0f));

        if (can_edit_any_pivot) {
            ImGui::SameLine();
            if (iconTypeButton("##center_pivot", ImVec2(btn_h, btn_h), ImVec4(0.95f, 0.72f, 0.28f, 1.0f), UIWidgets::IconType::PivotCenter, 20.0f)) {
                if (can_edit_object_pivot) {
                    std::string nm = sel.selected.object->nodeName;
                    if (nm.empty()) nm = "Unnamed";
                    recenterObjectPivotToBoundsCenter(ctx, nm);
                } else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) {
                    AABB bounds = sel.selected.vdb_volume->getWorldBounds();
                    Vec3 local = sel.selected.vdb_volume->getTransform().inverse().transform_point((bounds.min + bounds.max) * 0.5f);
                    sel.selected.vdb_volume->setPivotOffset(local);
                    SceneUI::syncVDBVolumesToGPU(ctx);
                } else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
                    Vec3 bmin, bmax;
                    sel.selected.gas_volume->getWorldBounds(bmin, bmax);
                    Vec3 local = sel.selected.gas_volume->getTransform().inverse().transform_point((bmin + bmax) * 0.5f);
                    sel.selected.gas_volume->setPivotOffset(local);
                    ctx.renderer.updateBackendGasVolumes(ctx.scene);
                }
                pivot_edit_mode = false;
                addViewportMessage("Pivot centered", 1.6f, ImVec4(1.0f, 0.8f, 0.35f, 1.0f));
                ProjectManager::getInstance().markModified();
            }
            viewportTooltip("Center Pivot", "Place the pivot at the selection center.", ImVec4(0.95f, 0.72f, 0.28f, 1.0f));
        } else {
            pivot_edit_mode = false;
        }
    }

    // ── Separator ──
    ImGui::SameLine(0, 6.0f);
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
    ImGui::SameLine(0, 6.0f);

    // ── Mouse Sensitivity (popup slider) ──
    {
        static bool sens_open = false;
        if (iconToggleTypeBtn("##toggle_sens", sens_open, ImVec2(btn_h + 2.0f, btn_h), ImVec4(0.62f, 0.58f, 0.24f, 1.0f), UIWidgets::IconType::Sensitivity, 21.0f)) {
            sens_open = !sens_open;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.92f, 0.84f, 0.30f, 1.0f));
            ImGui::TextUnformatted("Sensitivity");
            ImGui::PopStyleColor();
            ImGui::Separator();
            ImGui::Text("Current: %.3f", ctx.render_settings.mouse_sensitivity);
            ImGui::EndTooltip();
        }

        if (sens_open) {
            const float popup_w = 190.0f;
            float popup_x = ImGui::GetItemRectMax().x - popup_w;
            popup_x = (std::max)(8.0f, (std::min)(popup_x, io.DisplaySize.x - popup_w - 8.0f));
            const float popup_y = (std::min)(ImGui::GetItemRectMax().y + 6.0f, io.DisplaySize.y - 90.0f);
            ImGui::SetNextWindowPos(ImVec2(popup_x, popup_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(popup_w, 0));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.11f, 0.13f, 0.92f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
            if (ImGui::Begin("##SensPopup", &sens_open,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoFocusOnAppearing)) {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Sensitivity");
                ImGui::SetNextItemWidth(-1);
                ImGui::SliderFloat("##sens", &ctx.render_settings.mouse_sensitivity, 0.001f, 5.0f, "%.3f");
            }
            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        }
    }

    ImGui::PopStyleVar(4);  // WindowBorderSize, ItemSpacing, WindowRounding, WindowPadding
    ImGui::PopStyleColor(2); // Border, WindowBg
    ImGui::End();
    // overlay handled by raster grid (depth-tested) in the Vulkan backend

    // ── ViewCube: standard-view navigator (top-right, below the toolbar) ──
    // Click a face to snap to Top/Front/Side (orthographic); drag to free-orbit.
    if (ctx.scene.camera) {
        Camera& cam = *ctx.scene.camera;

        const float cube_sz = 92.0f;
        const float cube_x = io.DisplaySize.x - right_margin - cube_sz - 2.0f;
        const float cube_y = top_margin + 40.0f;

        ImGui::SetNextWindowPos(ImVec2(cube_x, cube_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(cube_sz, cube_sz), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGuiWindowFlags cubeFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoBackground;
        if (ImGui::Begin("##ViewCube", nullptr, cubeFlags)) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 wpos = ImGui::GetWindowPos();
            const ImVec2 wsz = ImGui::GetWindowSize();
            const ImVec2 center(wpos.x + wsz.x * 0.5f, wpos.y + wsz.y * 0.5f);
            const float radius = cube_sz * 0.30f;

            ImGui::SetCursorScreenPos(wpos);
            ImGui::InvisibleButton("##viewcube_hit", wsz);
            const bool hovered = ImGui::IsItemHovered();
            const bool active = ImGui::IsItemActive();

            // Camera view basis: right=u, up=v, toward-camera=w (maintained by update_camera_vectors).
            const Vec3 U = cam.u, Vv = cam.v, W = cam.w;
            auto project = [&](const Vec3& d) -> ImVec2 {
                return ImVec2(center.x + (float)Vec3::dot(d, U) * radius,
                              center.y - (float)Vec3::dot(d, Vv) * radius);
            };

            struct Face { Vec3 n, t0, t1; const char* label; Camera::StandardView view; };
            const Face faces[6] = {
                { Vec3( 1,0,0), Vec3(0,1,0), Vec3(0,0,1), "R",   Camera::StandardView::Right  },
                { Vec3(-1,0,0), Vec3(0,1,0), Vec3(0,0,1), "L",   Camera::StandardView::Left   },
                { Vec3(0, 1,0), Vec3(1,0,0), Vec3(0,0,1), "TOP", Camera::StandardView::Top    },
                { Vec3(0,-1,0), Vec3(1,0,0), Vec3(0,0,1), "BOT", Camera::StandardView::Bottom },
                { Vec3(0,0, 1), Vec3(1,0,0), Vec3(0,1,0), "F",   Camera::StandardView::Front  },
                { Vec3(0,0,-1), Vec3(1,0,0), Vec3(0,1,0), "BK",  Camera::StandardView::Back   },
            };

            ImVec2 quad[6][4];
            float fz[6];
            for (int i = 0; i < 6; ++i) {
                fz[i] = (float)Vec3::dot(faces[i].n, W);
                const Vec3 c = faces[i].n;
                quad[i][0] = project(c - faces[i].t0 - faces[i].t1);
                quad[i][1] = project(c + faces[i].t0 - faces[i].t1);
                quad[i][2] = project(c + faces[i].t0 + faces[i].t1);
                quad[i][3] = project(c - faces[i].t0 + faces[i].t1);
            }
            auto pointInQuad = [](const ImVec2* q, const ImVec2& p) -> bool {
                bool in = false;
                for (int a = 0, b = 3; a < 4; b = a++) {
                    if (((q[a].y > p.y) != (q[b].y > p.y)) &&
                        (p.x < (q[b].x - q[a].x) * (p.y - q[a].y) / (q[b].y - q[a].y) + q[a].x))
                        in = !in;
                }
                return in;
            };

            const ImVec2 mp = io.MousePos;
            int hoverFace = -1; float bestZ = 0.02f;
            if (hovered) {
                for (int i = 0; i < 6; ++i)
                    if (fz[i] > bestZ && pointInQuad(quad[i], mp)) { hoverFace = i; bestZ = fz[i]; }
            }

            int order[6] = { 0,1,2,3,4,5 };
            std::sort(order, order + 6, [&](int a, int b) { return fz[a] < fz[b]; });
            for (int oi = 0; oi < 6; ++oi) {
                const int i = order[oi];
                if (fz[i] <= 0.02f) continue; // back-facing
                const float shade = 0.30f + 0.45f * std::clamp(fz[i], 0.0f, 1.0f);
                const ImU32 fill = (i == hoverFace)
                    ? IM_COL32(90, 150, 230, 235)
                    : IM_COL32((int)(shade * 120), (int)(shade * 135), (int)(shade * 160), 210);
                dl->AddQuadFilled(quad[i][0], quad[i][1], quad[i][2], quad[i][3], fill);
                dl->AddQuad(quad[i][0], quad[i][1], quad[i][2], quad[i][3], IM_COL32(18, 22, 30, 220), 1.5f);
                const ImVec2 lc = project(faces[i].n);
                const ImVec2 ts = ImGui::CalcTextSize(faces[i].label);
                dl->AddText(ImVec2(lc.x - ts.x * 0.5f, lc.y - ts.y * 0.5f), IM_COL32(236, 239, 246, 255), faces[i].label);
            }

            // Refresh the interactive (solid/matcap) viewport. Push the camera straight to the
            // active shading backend + reset accumulation so it re-renders THIS frame — the
            // g_camera_dirty signal alone races with other consumers and was unreliable here.
            auto refreshViewport = [&]() {
                if (ctx.backend_ptr) {
                    ctx.backend_ptr->syncCamera(cam);
                    ctx.backend_ptr->resetAccumulation();
                }
                ctx.renderer.resetCPUAccumulation();
                extern bool g_camera_dirty;
                g_camera_dirty = true; // also drive the render-backend sync path next frame
                ProjectManager::getInstance().markModified();
            };

            // Orbit/snap pivot: selection centre when something is selected, else world origin
            // (where the reference grid lives) — keeps the view centred and the grid in frame.
            const Vec3 snapPivot = ctx.selection.hasSelection()
                ? ctx.selection.selected.position : Vec3(0.0f, 0.0f, 0.0f);

            // Click snaps to a standard view; drag (>4px) free-orbits around the pivot.
            static bool dragging = false; static bool moved = false; static ImVec2 dragLast;
            if (ImGui::IsItemActivated()) { dragging = true; moved = false; dragLast = mp; }
            if (active && dragging) {
                const ImVec2 d(mp.x - dragLast.x, mp.y - dragLast.y);
                if (std::abs(d.x) + std::abs(d.y) > 4.0f) moved = true;
                if (moved && (d.x != 0.0f || d.y != 0.0f)) {
                    const Vec3 pivot = cam.lookat;
                    Vec3 rel = cam.lookfrom - pivot;
                    auto rot = [](Vec3 p, Vec3 axis, float ang) {
                        axis = axis.normalize();
                        const float c = cosf(ang), s = sinf(ang);
                        return p * c + Vec3::cross(axis, p) * s + axis * ((float)Vec3::dot(axis, p)) * (1.0f - c);
                    };
                    const float k = 0.01f;
                    rel = rot(rel, Vec3(0, 1, 0), -d.x * k);
                    rel = rot(rel, cam.u, -d.y * k);
                    cam.lookfrom = pivot + rel;
                    // Free-orbiting the cube leaves the aligned ortho views and returns to a
                    // normal perspective camera (this is the way back out of orthographic).
                    cam.standard_view = Camera::StandardView::Perspective;
                    cam.orthographic = false;
                    cam.update_camera_vectors();
                    cam.markDirty();
                    refreshViewport();
                    dragLast = mp;
                }
            }
            if (ImGui::IsItemDeactivated()) {
                if (!moved && hoverFace >= 0) {
                    float dist = (cam.lookfrom - cam.lookat).length();
                    if (dist < 1e-3f) dist = 10.0f;
                    cam.setStandardView(faces[hoverFace].view, snapPivot, dist, true);
                    refreshViewport();
                    addViewportMessage("View aligned", 1.2f, ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
                }
                dragging = false;
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SPLIT-PRISM FOCUS INDICATOR (Classic SLR Style with Focus Ring)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawFocusIndicator(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show when DOF is enabled
    float aperture = ctx.scene.camera->aperture;
    if (aperture < 0.001f) return;  // DOF disabled

    // Only show if the toggles are enabled
    if (!viewport_settings.show_camera_hud) return;
    if (!viewport_settings.show_focus_ring) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Use Display dimensions for Foreground Overlay
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Indicator sizes
    float inner_radius = 25.0f;      // Split-prism area
    float outer_radius = 35.0f;      // Outer boundary
    float ring_radius = 50.0f;       // Focus ring (draggable)
    float ring_thickness = 12.0f;    // Focus ring width

    Camera& cam = *ctx.scene.camera;
    float& focus_dist = ctx.scene.camera->focus_dist;

    // Static state for dragging
    static bool is_dragging_ring = false;
    static float drag_start_x = 0.0f;
    static float drag_start_focus = 0.0f;

    // Colors
    ImU32 col_focused = IM_COL32(100, 255, 100, 200);    // Green when in focus
    ImU32 col_unfocused = IM_COL32(255, 150, 80, 180);   // Orange when out of focus
    ImU32 col_ring = IM_COL32(255, 255, 255, 120);       // White ring
    ImU32 col_ring_hover = IM_COL32(255, 255, 255, 200); // Bright when hovered
    ImU32 col_ring_active = IM_COL32(100, 200, 255, 255); // Blue when dragging
    ImU32 col_bg = IM_COL32(0, 0, 0, 60);                // Dark background
    ImU32 col_ring_bg = IM_COL32(40, 40, 40, 180);       // Ring background
    ImU32 col_tick = IM_COL32(200, 200, 200, 150);       // Tick marks

    // Check if mouse is over the focus ring
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - cx) * (mouse.x - cx) + (mouse.y - cy) * (mouse.y - cy));
    bool is_over_ring = (mouse_dist >= ring_radius - ring_thickness * 0.5f) &&
        (mouse_dist <= ring_radius + ring_thickness * 0.5f);

    // Handle focus ring dragging
    if (is_over_ring && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_ring = true;
        drag_start_x = mouse.x;
        drag_start_focus = focus_dist;
        hud_captured_mouse = true; // Prevent viewport selection
        
        // Auto-switch to MF when manually checking focus
        viewport_settings.focus_mode = 0;
    }

    if (is_dragging_ring) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Calculate focus change based on horizontal mouse movement
            float delta_x = mouse.x - drag_start_x;
            float sensitivity = 0.02f;  // Focus distance change per pixel

            focus_dist = drag_start_focus + delta_x * sensitivity;
            focus_dist = std::max(0.1f, std::min(focus_dist, 100.0f));  // Clamp

            cam.update_camera_vectors();
            ProjectManager::getInstance().markModified();

            // Update GPU
            if (ctx.backend_ptr) {
                ctx.renderer.syncCameraToBackend(*ctx.scene.camera);
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
        }
        else {
            is_dragging_ring = false;
        }
    }

    // Determine ring color
    ImU32 ring_col = col_ring;
    if (is_dragging_ring) {
        ring_col = col_ring_active;
    }
    else if (is_over_ring) {
        ring_col = col_ring_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW FOCUS RING (Lens barrel style)
    // ─────────────────────────────────────────────────────────────────────────

    // Ring background
    draw_list->AddCircle(ImVec2(cx, cy), ring_radius, col_ring_bg, 48, ring_thickness);

    // Ring with grip texture (tick marks)
    int num_ticks = 36;
    for (int i = 0; i < num_ticks; i++) {
        float angle = (2.0f * 3.14159f * i / num_ticks);
        float tick_inner = ring_radius - ring_thickness * 0.4f;
        float tick_outer = ring_radius + ring_thickness * 0.4f;

        // Alternating tick lengths
        if (i % 3 == 0) {
            tick_inner = ring_radius - ring_thickness * 0.3f;
            tick_outer = ring_radius + ring_thickness * 0.3f;
        }

        float x1 = cx + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = cx + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), ring_col, 1.5f);
    }

    // Highlight marker on ring (shows current focus "position")
    float focus_angle = -3.14159f * 0.5f + (focus_dist / 20.0f) * 3.14159f;  // Map 0-20m to angle
    float marker_x = cx + ring_radius * cosf(focus_angle);
    float marker_y = cy + ring_radius * sinf(focus_angle);
    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 5.0f, col_ring_active, 12);

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW SPLIT-PRISM INDICATOR (inside the ring)
    // ─────────────────────────────────────────────────────────────────────────

    // Inner circle background  
    draw_list->AddCircleFilled(ImVec2(cx, cy), outer_radius + 2, col_bg, 32);
    draw_list->AddCircle(ImVec2(cx, cy), outer_radius, col_ring, 32, 1.5f);

    // Calculate focus accuracy
    float offset = 0.0f;
    ImU32 indicator_col = col_focused;
    float normalized_error = 0.0f;
    float obj_dist = focus_dist;  // Default to focus dist if no selection

    if (ctx.selection.hasSelection()) {
        Vec3 obj_pos = ctx.selection.selected.position;
        Vec3 cam_pos = cam.lookfrom;
        obj_dist = (obj_pos - cam_pos).length();

        float focus_error = std::abs(obj_dist - focus_dist);
        normalized_error = focus_error / (focus_dist + 0.01f);

        // Calculate visual offset (max 12 pixels) - more sensitive
        offset = std::min(normalized_error * 40.0f, 12.0f);

        // Color based on focus accuracy - tighter tolerance (5%)
        if (normalized_error < 0.05f) {
            indicator_col = col_focused;  // Green - in focus
            offset = 0.0f;  // Perfect alignment when in focus
        }
        else if (normalized_error < 0.15f) {
            // Blend green to orange
            float t = (normalized_error - 0.05f) / 0.10f;
            int r = (int)(100 + (255 - 100) * t);
            int g = (int)(255 + (150 - 255) * t);
            int b = (int)(100 + (80 - 100) * t);
            indicator_col = IM_COL32(r, g, b, 200);
        }
        else {
            indicator_col = col_unfocused;  // Orange - out of focus
        }
    }

    // Draw split circles
    ImVec2 top_center(cx, cy - offset);
    ImVec2 bot_center(cx, cy + offset);
    float line_thickness = 2.5f;

    // Top half - semicircle
    draw_list->PathClear();
    for (int i = 0; i <= 16; i++) {
        float angle = 3.14159f + (3.14159f * i / 16.0f);
        float x = top_center.x + inner_radius * cosf(angle);
        float y = top_center.y + inner_radius * sinf(angle);
        draw_list->PathLineTo(ImVec2(x, y));
    }
    draw_list->PathStroke(indicator_col, 0, line_thickness);

    // Bottom half - semicircle
    draw_list->PathClear();
    for (int i = 0; i <= 16; i++) {
        float angle = (3.14159f * i / 16.0f);
        float x = bot_center.x + inner_radius * cosf(angle);
        float y = bot_center.y + inner_radius * sinf(angle);
        draw_list->PathLineTo(ImVec2(x, y));
    }
    draw_list->PathStroke(indicator_col, 0, line_thickness);

    // Center crosshair
    float cross_size = 5.0f;
    draw_list->AddLine(ImVec2(cx - cross_size, cy), ImVec2(cx + cross_size, cy), col_ring, 1.0f);
    draw_list->AddLine(ImVec2(cx, cy - cross_size), ImVec2(cx, cy + cross_size), col_ring, 1.0f);

    // ─────────────────────────────────────────────────────────────────────────
    // TEXT LABELS
    // ─────────────────────────────────────────────────────────────────────────

    // Focus distance text below
    char focus_text[32];
    snprintf(focus_text, sizeof(focus_text), "%.2fm", focus_dist);
    ImVec2 text_size = ImGui::CalcTextSize(focus_text);
    draw_list->AddText(ImVec2(cx - text_size.x * 0.5f, cy + ring_radius + ring_thickness * 0.5f + 5), col_ring, focus_text);

    // Status text above
    const char* status_text = "";
    if (ctx.selection.hasSelection()) {
        if (normalized_error < 0.05f) {
            status_text = "IN FOCUS";
        }
        else if (obj_dist < focus_dist) {
            status_text = "FRONT FOCUS";
        }
        else {
            status_text = "BACK FOCUS";
        }

        ImVec2 status_size = ImGui::CalcTextSize(status_text);
        draw_list->AddText(ImVec2(cx - status_size.x * 0.5f, cy - ring_radius - ring_thickness * 0.5f - 18), indicator_col, status_text);
    }

    // Drag hint when hovering ring
    if (is_over_ring && !is_dragging_ring) {
        const char* hint = "Drag to adjust focus";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(cx - hint_size.x * 0.5f, cy + ring_radius + ring_thickness * 0.5f + 22), col_ring_hover, hint);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ZOOM RING (FOV Control - Outer ring around focus indicator)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawZoomRing(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if the toggles are enabled
    if (!viewport_settings.show_camera_hud) return;
    if (!viewport_settings.show_zoom_ring) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;
    float& fov = ctx.scene.camera->vfov;

    // Center of viewport (same as focus ring)
    // Center of viewport (Foreground)
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Zoom ring is outside focus ring
    // Focus ring: inner=25, outer=35, ring_radius=50
    float zoom_ring_radius = 72.0f;      // Outside focus ring
    float zoom_ring_thickness = 10.0f;

    // Static state for dragging
    static bool is_dragging_zoom = false;
    static float drag_start_x = 0.0f;
    static float drag_start_fov = 0.0f;

    // Colors (blue tint to differentiate from focus ring)
    ImU32 col_ring = IM_COL32(100, 150, 200, 150);
    ImU32 col_ring_hover = IM_COL32(120, 180, 255, 200);
    ImU32 col_ring_active = IM_COL32(80, 150, 255, 255);
    ImU32 col_tick = IM_COL32(180, 200, 220, 150);
    ImU32 col_marker = IM_COL32(255, 200, 80, 255);
    ImU32 col_text = IM_COL32(200, 220, 255, 200);

    // Check if mouse is over the zoom ring
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - cx) * (mouse.x - cx) + (mouse.y - cy) * (mouse.y - cy));
    bool is_over_ring = (mouse_dist >= zoom_ring_radius - zoom_ring_thickness * 0.5f) &&
        (mouse_dist <= zoom_ring_radius + zoom_ring_thickness * 0.5f);

    // Handle zoom ring dragging
    if (is_over_ring && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_zoom = true;
        drag_start_x = mouse.x;
        drag_start_fov = fov;
        hud_captured_mouse = true; // Prevent viewport selection
    }

    if (is_dragging_zoom) {
        // Enforce Prime/Zoom constraints
        int lens_idx = cam.lens_preset_index;
        bool is_prime = false;
        float min_limit_mm = 12.0f;
        float max_limit_mm = 1200.0f;
        
        // Validation of index
        if (lens_idx > 0 && lens_idx < (int)CameraPresets::LENS_PRESET_COUNT) {
            const auto& preset = CameraPresets::LENS_PRESETS[lens_idx];
            if (!preset.is_zoom) {
                is_prime = true;
            } else {
                min_limit_mm = preset.min_mm;
                max_limit_mm = preset.max_mm;
            }
        }
        
        // Disable drag if Prime lens
        if (is_prime) {
             is_dragging_zoom = false; // Stop dragging immediately
             // Ideally show a toast/notification "Prime Lens - Fixed Focal Length"
             addViewportMessage("Prime Lens: Fixed Focal Length", 1.0f, ImVec4(1, 0.5f, 0, 1));
        }
        else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            float delta_x = mouse.x - drag_start_x;
            
            // Calculate current focal length from drag_start_fov
            float start_focal = 24.0f / (2.0f * tanf(drag_start_fov * 0.5f * 3.14159f / 180.0f));
            
            // Adjust sensitivity based on focal length
            float sensitivity = 0.15f;  
            if (start_focal > 300.0f) sensitivity = 0.02f; 
            else if (start_focal > 100.0f) sensitivity = 0.05f;

            fov = drag_start_fov + delta_x * sensitivity;
            
            // Clamp to active lens capacity (or global limits if Custom)
            float max_fov_limit = 2.0f * atanf(24.0f / (2.0f * min_limit_mm)) * 180.0f / 3.14159f;
            float min_fov_limit = 2.0f * atanf(24.0f / (2.0f * max_limit_mm)) * 180.0f / 3.14159f;
            
            fov = std::max(min_fov_limit, std::min(fov, max_fov_limit));

            ctx.scene.camera->fov = fov;
            cam.update_camera_vectors();

            // Update GPU
            if (ctx.backend_ptr) {
                ctx.renderer.syncCameraToBackend(*ctx.scene.camera);
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        else {
            is_dragging_zoom = false;
        }
    }

    // Determine ring color
    ImU32 ring_col = col_ring;
    if (is_dragging_zoom) {
        ring_col = col_ring_active;
    }
    else if (is_over_ring) {
        ring_col = col_ring_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW ZOOM RING
    // ─────────────────────────────────────────────────────────────────────────

    // Ring with grip texture (tick marks)
    int num_ticks = 48;
    for (int i = 0; i < num_ticks; i++) {
        float angle = (2.0f * 3.14159f * i / num_ticks);
        float tick_inner = zoom_ring_radius - zoom_ring_thickness * 0.4f;
        float tick_outer = zoom_ring_radius + zoom_ring_thickness * 0.4f;

        // Alternating tick lengths
        if (i % 4 == 0) {
            tick_inner = zoom_ring_radius - zoom_ring_thickness * 0.3f;
            tick_outer = zoom_ring_radius + zoom_ring_thickness * 0.3f;
        }

        float x1 = cx + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = cx + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), ring_col, 1.0f);
    }

    // Current FOV marker position (maps FOV 1.0-120 to angle)
    float fov_t = (fov - 1.0f) / 119.0f;  // 0 to 1 (Range: 1 to 120)
    float fov_angle = -3.14159f * 0.5f + fov_t * 3.14159f;  // -90° to +90°
    float marker_x = cx + zoom_ring_radius * cosf(fov_angle);
    float marker_y = cy + zoom_ring_radius * sinf(fov_angle);

    // Draw marker (larger, more visible)
    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 6.0f, col_marker, 12);
    draw_list->AddCircle(ImVec2(marker_x, marker_y), 6.0f, IM_COL32(0, 0, 0, 150), 12, 1.5f);

    // ─────────────────────────────────────────────────────────────────────────
    // FOV LABELS (corners of the ring arc)
    // ─────────────────────────────────────────────────────────────────────────

    // Wide label (left side)
    float wide_angle = -3.14159f * 0.5f;
    float wide_x = cx + (zoom_ring_radius + 18) * cosf(wide_angle);
    float wide_y = cy + (zoom_ring_radius + 18) * sinf(wide_angle);
    draw_list->AddText(ImVec2(wide_x - 10, wide_y - 6), col_tick, "W");

    // Tele label (right side)
    float tele_angle = 3.14159f * 0.5f;
    float tele_x = cx + (zoom_ring_radius + 18) * cosf(tele_angle);
    float tele_y = cy + (zoom_ring_radius + 18) * sinf(tele_angle);
    draw_list->AddText(ImVec2(tele_x - 2, tele_y - 6), col_tick, "T");

    // Current FOV/focal text (bottom)
    char fov_text[32];
    float focal_mm = 24.0f / (2.0f * tanf(fov * 0.5f * 3.14159f / 180.0f));
    snprintf(fov_text, sizeof(fov_text), "%.0f deg / %.0fmm", fov, focal_mm);
    ImVec2 fov_size = ImGui::CalcTextSize(fov_text);
    draw_list->AddText(ImVec2(cx - fov_size.x * 0.5f, cy + zoom_ring_radius + zoom_ring_thickness * 0.5f + 24), col_text, fov_text);
}

// ═════════════════════════════════════════════════════════════════════════════
// DOLLY ARC (Camera Track Control - Left side arc)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawDollyArc(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if Camera HUD is enabled
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;

    // Position: Left side of center (Foreground)
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Arc properties - left side of focus ring
    float arc_radius = 72.0f;           // Same as zoom ring
    float arc_thickness = 10.0f;
    float arc_center_x = cx - 0;        // Offset left from center

    // Arc spans from 90° to 270° (left half circle)
    float arc_start = 3.14159f * 0.5f;   // 90°
    float arc_end = 3.14159f * 1.5f;     // 270°

    // Colors
    ImU32 col_arc = IM_COL32(180, 120, 80, 150);      // Warm brown for dolly (different from blue zoom)
    ImU32 col_arc_hover = IM_COL32(220, 160, 100, 200);
    ImU32 col_arc_active = IM_COL32(255, 200, 120, 255);
    ImU32 col_marker = IM_COL32(255, 180, 80, 255);
    ImU32 col_text = IM_COL32(220, 180, 140, 220);

    // Check if mouse is over the left arc
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - arc_center_x) * (mouse.x - arc_center_x) + (mouse.y - cy) * (mouse.y - cy));
    float mouse_angle = atan2f(mouse.y - cy, mouse.x - arc_center_x);
    if (mouse_angle < 0) mouse_angle += 2.0f * 3.14159f;

    bool is_in_arc_ring = (mouse_dist >= arc_radius - arc_thickness * 0.5f) &&
        (mouse_dist <= arc_radius + arc_thickness * 0.5f);
    bool is_in_arc_angle = (mouse_angle >= arc_start && mouse_angle <= arc_end) ||
        (mouse.x < arc_center_x);  // Left side
    bool is_over_arc = is_in_arc_ring && is_in_arc_angle;

    // Dragging state
    static bool is_dragging_dolly = false;
    static float drag_start_x = 0.0f;
    static float drag_start_dolly = 0.0f;
    static Vec3 dolly_initial_pos;

    // Handle dolly arc dragging
    if (is_over_arc && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_dolly = true;
        drag_start_x = mouse.x;
        drag_start_dolly = cam.dolly_position;
        dolly_initial_pos = cam.lookfrom;

        // Set rig mode to Dolly
        cam.rig_mode = Camera::RigMode::Dolly;
        cam.dolly_start_pos = cam.lookfrom;
        hud_captured_mouse = true; // Prevent viewport selection
    }

    if (is_dragging_dolly) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Calculate dolly movement based on horizontal drag
            float delta_x = (mouse.x - drag_start_x) * 0.02f * cam.dolly_speed;
            cam.dolly_position = drag_start_dolly + delta_x;

            // Move camera along its right vector
            Vec3 right = cam.u;  // Camera right vector
            Vec3 new_pos = dolly_initial_pos + right * delta_x;
            cam.lookfrom = new_pos;
            cam.lookat = cam.lookat + right * delta_x;  // Keep looking at same relative point
            cam.update_camera_vectors();

            // Update GPU
            if (ctx.backend_ptr) {
                ctx.renderer.syncCameraToBackend(cam);
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        else {
            is_dragging_dolly = false;
        }
    }

    // Determine arc color
    ImU32 arc_col = col_arc;
    if (is_dragging_dolly) {
        arc_col = col_arc_active;
    }
    else if (is_over_arc) {
        arc_col = col_arc_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW DOLLY ARC (Left side, with tick marks)
    // ─────────────────────────────────────────────────────────────────────────

    // Draw arc with tick marks
    int num_ticks = 24;  // Half of zoom ring's ticks
    for (int i = 0; i < num_ticks; i++) {
        float t = (float)i / (num_ticks - 1);
        float angle = arc_start + t * (arc_end - arc_start);

        float tick_inner = arc_radius - arc_thickness * 0.4f;
        float tick_outer = arc_radius + arc_thickness * 0.4f;

        if (i % 4 == 0) {
            tick_inner = arc_radius - arc_thickness * 0.3f;
            tick_outer = arc_radius + arc_thickness * 0.3f;
        }

        float x1 = arc_center_x + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = arc_center_x + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), arc_col, 1.0f);
    }

    // Draw dolly position marker (maps dolly_position -5 to +5 across arc)
    float dolly_clamped = std::max(-5.0f, std::min(cam.dolly_position, 5.0f));
    float dolly_t = (dolly_clamped + 5.0f) / 10.0f;
    float marker_angle = arc_start + dolly_t * (arc_end - arc_start);
    float marker_x = arc_center_x + arc_radius * cosf(marker_angle);
    float marker_y = cy + arc_radius * sinf(marker_angle);

    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 6.0f, col_marker, 12);
    draw_list->AddCircle(ImVec2(marker_x, marker_y), 6.0f, IM_COL32(0, 0, 0, 150), 12, 1.5f);

    // ─────────────────────────────────────────────────────────────────────────
    // LABELS
    // ─────────────────────────────────────────────────────────────────────────

    // "DOLLY" label at top of arc
    float label_angle = 3.14159f;  // 180° (left)
    float label_x = arc_center_x + (arc_radius + 18) * cosf(label_angle);
    float label_y = cy + (arc_radius + 18) * sinf(label_angle);
    draw_list->AddText(ImVec2(label_x - 22, label_y - 6), col_text, "DOLLY");

    // Position value
    char pos_text[16];
    snprintf(pos_text, sizeof(pos_text), "%.1fm", cam.dolly_position);
    ImVec2 pos_size = ImGui::CalcTextSize(pos_text);
    draw_list->AddText(ImVec2(arc_center_x - arc_radius - 35, cy - pos_size.y * 0.5f), col_marker, pos_text);

    // Drag hint
    if (is_over_arc && !is_dragging_dolly) {
        const char* hint = "Drag to dolly";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(arc_center_x - arc_radius - 40, cy + 20), col_arc_hover, hint);
    }

    // Active indicator
    if (cam.rig_mode == Camera::RigMode::Dolly) {
        draw_list->AddText(ImVec2(arc_center_x - arc_radius - 30, cy - 35), col_arc_active, "[ACTIVE]");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawExposureInfo(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if Camera HUD is enabled
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;

    // Read ISO from preset
    int& iso_idx = cam.iso_preset_index;
    if (iso_idx < 0) iso_idx = 0;
    if (iso_idx >= (int)CameraPresets::ISO_PRESET_COUNT) iso_idx = (int)CameraPresets::ISO_PRESET_COUNT - 1;
    int iso = CameraPresets::ISO_PRESETS[iso_idx].iso_value;

    // Read Shutter from preset  
    int& shutter_idx = cam.shutter_preset_index;
    if (shutter_idx < 0) shutter_idx = 0;
    if (shutter_idx >= (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) shutter_idx = (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT - 1;
    float shutter_seconds = CameraPresets::SHUTTER_SPEED_PRESETS[shutter_idx].speed_seconds;

    // Get f-stop from fstop preset
    int& fstop_idx = cam.fstop_preset_index;
    if (fstop_idx < 0) fstop_idx = 0;
    if (fstop_idx >= (int)CameraPresets::FSTOP_PRESET_COUNT) fstop_idx = (int)CameraPresets::FSTOP_PRESET_COUNT - 1;
    float f_stop = CameraPresets::FSTOP_PRESETS[fstop_idx].f_number;

    // Calculate EV (Exposure Value) - Correct formula
    // EV = log2(N² / t) where N = f-number, t = shutter time in seconds
    // For ISO: EV_100 = EV + log2(ISO/100)
    float ev100 = log2f((f_stop * f_stop) / shutter_seconds);
    float ev = ev100 - log2f((float)iso / 100.0f);

    // Clamp EV to reasonable range
    ev = std::max(-6.0f, std::min(ev, 20.0f));

    // Position: Right side, middle height (Foreground)
    float margin_right = 30.0f;
    float triangle_size = 85.0f;

    float cx = io.DisplaySize.x - margin_right - triangle_size * 0.7f;
    float cy = io.DisplaySize.y * 0.30f;

    // Colors - very transparent background, brighter text
    ImU32 col_bg = IM_COL32(0, 0, 0, 80);             // Very transparent
    ImU32 col_border = IM_COL32(200, 200, 200, 150);
    ImU32 col_label = IM_COL32(255, 255, 255, 255);  // White labels
    ImU32 col_value = IM_COL32(255, 220, 100, 255);  // Bright amber
    ImU32 col_value_hover = IM_COL32(100, 200, 255, 255);  // Blue when hovering
    ImU32 col_ev_positive = IM_COL32(255, 150, 80, 255);
    ImU32 col_ev_negative = IM_COL32(80, 150, 255, 255);
    ImU32 col_ev_neutral = IM_COL32(100, 255, 100, 255);

    // Triangle vertices
    float height = triangle_size * 0.866f;
    ImVec2 v_iso(cx, cy - height * 0.5f);
    ImVec2 v_shutter(cx - triangle_size * 0.5f, cy + height * 0.4f);
    ImVec2 v_aperture(cx + triangle_size * 0.5f, cy + height * 0.4f);

    // Draw triangle background
    draw_list->AddTriangleFilled(v_iso, v_shutter, v_aperture, col_bg);
    draw_list->AddTriangle(v_iso, v_shutter, v_aperture, col_border, 1.5f);

    // Check mouse position for interactivity
    ImVec2 mouse = io.MousePos;
    static int dragging = 0;  // 0=none, 1=ISO, 2=Shutter, 3=Aperture
    static float drag_start_x = 0;
    static int drag_start_idx = 0;

    // Hitbox rectangles for each control
    float hit_size = 35.0f;
    bool iso_hover = (mouse.x >= v_iso.x - hit_size && mouse.x <= v_iso.x + hit_size &&
        mouse.y >= v_iso.y - 30 && mouse.y <= v_iso.y + 15);
    bool shutter_hover = (mouse.x >= v_shutter.x - hit_size && mouse.x <= v_shutter.x + hit_size &&
        mouse.y >= v_shutter.y && mouse.y <= v_shutter.y + 35);
    bool aperture_hover = (mouse.x >= v_aperture.x - hit_size && mouse.x <= v_aperture.x + hit_size &&
        mouse.y >= v_aperture.y && mouse.y <= v_aperture.y + 35);

    // Handle dragging
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (iso_hover) { dragging = 1; drag_start_x = mouse.x; drag_start_idx = iso_idx; hud_captured_mouse = true; }
            else if (shutter_hover) { dragging = 2; drag_start_x = mouse.x; drag_start_idx = shutter_idx; hud_captured_mouse = true; }
            else if (aperture_hover) { dragging = 3; drag_start_x = mouse.x; drag_start_idx = fstop_idx; hud_captured_mouse = true; }
        }
    }

    if (dragging > 0 && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float delta = (mouse.x - drag_start_x) / 20.0f;
        int new_idx = drag_start_idx + (int)delta;

        bool value_changed = false;

        if (dragging == 1) {
            int old_idx = iso_idx;
            iso_idx = std::max(0, std::min(new_idx, (int)CameraPresets::ISO_PRESET_COUNT - 1));
            if (iso_idx != old_idx) {
                cam.iso_preset_index = iso_idx;
                int iso_val = CameraPresets::ISO_PRESETS[iso_idx].iso_value;
                cam.iso = iso_val;
                value_changed = true;
            }
        }
        else if (dragging == 2) {
            int old_idx = shutter_idx;
            shutter_idx = std::max(0, std::min(new_idx, (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT - 1));
            if (shutter_idx != old_idx) {
                cam.shutter_preset_index = shutter_idx;
                cam.shutter_speed = 1.0f / CameraPresets::SHUTTER_SPEED_PRESETS[shutter_idx].speed_seconds;
                value_changed = true;
            }
        }
        else if (dragging == 3) {
            int old_idx = fstop_idx;
            fstop_idx = std::max(0, std::min(new_idx, (int)CameraPresets::FSTOP_PRESET_COUNT - 1));
            if (fstop_idx != old_idx) {
                // Update actual aperture and lens_radius from f-stop preset
                float f_val = CameraPresets::FSTOP_PRESETS[fstop_idx].f_number;
                cam.aperture = CameraPresets::FSTOP_PRESETS[fstop_idx].aperture_value;
                cam.lens_radius = cam.aperture * 0.5f;
                cam.fstop_preset_index = fstop_idx; // Synch preset index
                value_changed = true;
            }
        }

        // When manually changing exposure, disable auto and enable physical exposure
        if (value_changed) {
            cam.auto_exposure = false;  // Disable auto exposure
            cam.use_physical_exposure = true;  // Enable physical calculation (GPU parity)

            // Update GPU and reset render
            if (ctx.backend_ptr) {
                ctx.renderer.syncCameraToBackend(cam);
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();

            // Set warning message timer
            static float warning_timer = 0.0f;
            warning_timer = 3.0f;  // Show for 3 seconds
        }
    }
    else {
        dragging = 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AE (Auto Exposure) TOGGLE - Small button in center of triangle
    // ─────────────────────────────────────────────────────────────────────────
    const char* ae_text = "AE";
    ImVec2 ae_size = ImGui::CalcTextSize(ae_text);
    float ae_x = cx - ae_size.x * 0.5f;
    float ae_y = cy + 12;  // Just below EV

    // Hitbox for AE toggle
    bool ae_hover = (mouse.x >= ae_x - 8 && mouse.x <= ae_x + ae_size.x + 8 &&
        mouse.y >= ae_y - 4 && mouse.y <= ae_y + ae_size.y + 4);

    // Click to toggle auto exposure
    if (ae_hover && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        cam.auto_exposure = !cam.auto_exposure;
        hud_captured_mouse = true; // Prevent viewport selection

        if (ctx.backend_ptr) {
            ctx.renderer.syncCameraToBackend(cam);
            ctx.backend_ptr->resetAccumulation();
        }
        ctx.renderer.resetCPUAccumulation();
        ProjectManager::getInstance().markModified();
    }

    // Draw AE button - Green=ON, Red=OFF
    ImU32 ae_bg = cam.auto_exposure ? IM_COL32(50, 150, 50, 220) : IM_COL32(150, 50, 50, 220);
    ImU32 ae_col = ae_hover ? IM_COL32(255, 255, 255, 255) : IM_COL32(220, 220, 220, 255);

    draw_list->AddRectFilled(ImVec2(ae_x - 6, ae_y - 3), ImVec2(ae_x + ae_size.x + 6, ae_y + ae_size.y + 3), ae_bg, 4.0f);
    draw_list->AddText(ImVec2(ae_x, ae_y), ae_col, ae_text);

    // Show warning message only once when auto exposure is first disabled
    static bool was_auto_on = true;
    static float warning_timer = 0.0f;

    if (was_auto_on && !cam.auto_exposure) {
        warning_timer = 3.0f;  // Trigger warning
    }
    was_auto_on = cam.auto_exposure;

    if (warning_timer > 0.0f) {
        warning_timer -= io.DeltaTime;
        const char* warning = "Auto Exposure OFF";
        ImVec2 warn_size = ImGui::CalcTextSize(warning);
        float warn_x = cx - warn_size.x * 0.5f;
        float warn_y = cy + height * 0.5f + 30;

        // Fade out effect
        float alpha = std::min(warning_timer, 1.0f);
        ImU32 warn_bg = IM_COL32(0, 0, 0, (int)(180 * alpha));
        ImU32 warn_text = IM_COL32(255, 200, 50, (int)(255 * alpha));

        draw_list->AddRectFilled(ImVec2(warn_x - 5, warn_y - 2), ImVec2(warn_x + warn_size.x + 5, warn_y + warn_size.y + 2), warn_bg, 3.0f);
        draw_list->AddText(ImVec2(warn_x, warn_y), warn_text, warning);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ISO (Top vertex)
    // ─────────────────────────────────────────────────────────────────────────
    char iso_text[16];
    snprintf(iso_text, sizeof(iso_text), "%d", CameraPresets::ISO_PRESETS[iso_idx].iso_value);
    ImVec2 iso_size = ImGui::CalcTextSize(iso_text);
    ImU32 iso_col = (iso_hover || dragging == 1) ? col_value_hover : col_value;

    // Label with shadow for visibility
    draw_list->AddText(ImVec2(v_iso.x - 9, v_iso.y - 23), IM_COL32(0, 0, 0, 200), "ISO");
    draw_list->AddText(ImVec2(v_iso.x - 8, v_iso.y - 22), col_label, "ISO");
    draw_list->AddText(ImVec2(v_iso.x - iso_size.x * 0.5f + 1, v_iso.y - 7), IM_COL32(0, 0, 0, 200), iso_text);
    draw_list->AddText(ImVec2(v_iso.x - iso_size.x * 0.5f, v_iso.y - 8), iso_col, iso_text);

    // ─────────────────────────────────────────────────────────────────────────
    // SHUTTER (Bottom-left vertex)
    // ─────────────────────────────────────────────────────────────────────────
    const char* shutter_name = CameraPresets::SHUTTER_SPEED_PRESETS[shutter_idx].name;
    ImVec2 shutter_size = ImGui::CalcTextSize(shutter_name);
    ImU32 shutter_col = (shutter_hover || dragging == 2) ? col_value_hover : col_value;

    draw_list->AddText(ImVec2(v_shutter.x - 9, v_shutter.y + 3), IM_COL32(0, 0, 0, 200), "SH");
    draw_list->AddText(ImVec2(v_shutter.x - 8, v_shutter.y + 4), col_label, "SH");
    draw_list->AddText(ImVec2(v_shutter.x - shutter_size.x * 0.5f + 1, v_shutter.y + 17), IM_COL32(0, 0, 0, 200), shutter_name);
    draw_list->AddText(ImVec2(v_shutter.x - shutter_size.x * 0.5f, v_shutter.y + 16), shutter_col, shutter_name);

    // ─────────────────────────────────────────────────────────────────────────
    // APERTURE (Bottom-right vertex)
    // ─────────────────────────────────────────────────────────────────────────
    char aperture_text[16];
    snprintf(aperture_text, sizeof(aperture_text), "f/%.1f", CameraPresets::FSTOP_PRESETS[fstop_idx].f_number);
    ImVec2 aperture_size = ImGui::CalcTextSize(aperture_text);
    ImU32 aperture_col = (aperture_hover || dragging == 3) ? col_value_hover : col_value;

    draw_list->AddText(ImVec2(v_aperture.x - 1, v_aperture.y + 3), IM_COL32(0, 0, 0, 200), "AP");
    draw_list->AddText(ImVec2(v_aperture.x, v_aperture.y + 4), col_label, "AP");
    draw_list->AddText(ImVec2(v_aperture.x - aperture_size.x * 0.5f + 1, v_aperture.y + 17), IM_COL32(0, 0, 0, 200), aperture_text);
    draw_list->AddText(ImVec2(v_aperture.x - aperture_size.x * 0.5f, v_aperture.y + 16), aperture_col, aperture_text);

    // ─────────────────────────────────────────────────────────────────────────
    // EV (Center of triangle)
    // ─────────────────────────────────────────────────────────────────────────
    ImU32 ev_col = col_ev_neutral;
    if (ev > 12.0f) ev_col = col_ev_positive;  // Bright scene
    else if (ev < 8.0f) ev_col = col_ev_negative;  // Dark scene

    char ev_text[16];
    snprintf(ev_text, sizeof(ev_text), "EV %.0f", ev);
    ImVec2 ev_size = ImGui::CalcTextSize(ev_text);
    draw_list->AddText(ImVec2(cx - ev_size.x * 0.5f + 1, cy - 3), IM_COL32(0, 0, 0, 200), ev_text);
    draw_list->AddText(ImVec2(cx - ev_size.x * 0.5f, cy - 4), ev_col, ev_text);

    // Drag hint
    if (iso_hover || shutter_hover || aperture_hover) {
        const char* hint = "<< drag >>";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(cx - hint_size.x * 0.5f, cy + height * 0.5f + 10), col_value_hover, hint);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // LENS INFO (Below the triangle - transparent overlay)
    // ─────────────────────────────────────────────────────────────────────────
    float lens_y = cy + height * 0.5f + 35.0f;  // Moved up (was +50)
    
    // Calculate focal length from FOV
    float fov = (float)cam.vfov;
    float focal_mm = 24.0f / (2.0f * tanf(fov * 0.5f * 3.14159f / 180.0f));
    
    // Determine lens type name
    const char* lens_type = "";
    if (focal_mm < 20.0f) lens_type = "Ultra Wide";
    else if (focal_mm < 28.0f) lens_type = "Wide";
    else if (focal_mm < 40.0f) lens_type = "Std Wide";
    else if (focal_mm < 60.0f) lens_type = "Normal";
    else if (focal_mm < 100.0f) lens_type = "Portrait";
    else if (focal_mm < 200.0f) lens_type = "Tele";
    else if (focal_mm < 400.0f) lens_type = "Long Tele";
    else if (focal_mm < 800.0f) lens_type = "Super Tele";
    else lens_type = "Extreme";
    
    // Line 1: Focal length + type
    char lens_line1[48];
    snprintf(lens_line1, sizeof(lens_line1), "%.0fmm %s", focal_mm, lens_type);
    ImVec2 lens1_size = ImGui::CalcTextSize(lens_line1);
    
    // Semi-transparent colors
    ImU32 col_lens_label = IM_COL32(180, 200, 255, 180);
    ImU32 col_lens_value = IM_COL32(255, 230, 150, 200);
    
    // Shadow + text
    draw_list->AddText(ImVec2(cx - lens1_size.x * 0.5f + 1, lens_y + 1), IM_COL32(0, 0, 0, 150), lens_line1);
    draw_list->AddText(ImVec2(cx - lens1_size.x * 0.5f, lens_y), col_lens_value, lens_line1);
    
    // Line 2: Focus distance
    char lens_line2[32];
    snprintf(lens_line2, sizeof(lens_line2), "Focus: %.2fm", (float)cam.focus_dist);
    ImVec2 lens2_size = ImGui::CalcTextSize(lens_line2);
    
    // Increased spacing (+18px instead of +15/16) to prevent overlap
    draw_list->AddText(ImVec2(cx - lens2_size.x * 0.5f + 1, lens_y + 19), IM_COL32(0, 0, 0, 150), lens_line2);
    draw_list->AddText(ImVec2(cx - lens2_size.x * 0.5f, lens_y + 18), col_lens_label, lens_line2);
}

// ============================================================================
// VIEWPORT MESSAGES (HUD) - Display simple toast notifications
// ============================================================================

void SceneUI::addViewportMessage(const std::string& text, float duration, ImVec4 color) {
    // Check for duplicate message and update if found
    for (auto& msg : active_messages) {
        if (msg.text == text) {
            msg.time_remaining = duration; // Reset timer
            msg.color = color;             // Update color
            return;
        }
    }

    // Coalesce similar messages (e.g. per-frame "Render pass ..." or Vulkan warnings)
    // Compare prefixes to avoid HUD spam when only numeric/frame counters change.
    const size_t coalescePrefix = 24;
    for (auto& msg : active_messages) {
        size_t a = std::min(coalescePrefix, msg.text.size());
        size_t b = std::min(coalescePrefix, text.size());
        if (a > 0 && b > 0) {
            size_t n = std::min(a, b);
            if (msg.text.compare(0, n, text, 0, n) == 0) {
                // Considered similar; refresh and merge
                msg.time_remaining = std::max(msg.time_remaining, duration);
                msg.color = color;
                return;
            }
        }
    }

    ViewportMessage msg;
    msg.text = text;
    msg.time_remaining = duration;
    msg.color = color;
    active_messages.push_back(msg);
}

void SceneUI::clearViewportMessages() {
    active_messages.clear();
}

void SceneUI::drawViewportMessages(UIContext& ctx, float left_offset) {
    // ALWAYS draw if selection exists OR messages exist OR scene is initialized (for render stats)
    if (active_messages.empty() && !ctx.selection.hasSelection() && !ctx.scene.initialized) return;

    ImGuiIO& io = ImGui::GetIO();
    float dt = io.DeltaTime;

    const auto formatCompactCount = [](size_t value) -> std::string {
        char buffer[32];
        if (value >= 1000000ull) {
            std::snprintf(buffer, sizeof(buffer), "%.1fM", static_cast<double>(value) / 1000000.0);
        } else if (value >= 1000ull) {
            std::snprintf(buffer, sizeof(buffer), "%.1fK", static_cast<double>(value) / 1000.0);
        } else {
            std::snprintf(buffer, sizeof(buffer), "%zu", value);
        }
        return buffer;
    };
    
    // Position: Top Left of Viewport (respecting Left Panel)
    // Left Offset + 20px padding (user requested: "panelin solunda" but "viewport içinde", assuming next to panel)
    float x = left_offset + 20.0f;
    float y = 50.0f; // Below menu bar (approx)

    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f); // Invisible background (HUD style)
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | 
                             ImGuiWindowFlags_NoResize | 
                             ImGuiWindowFlags_AlwaysAutoResize | 
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoInputs; // Pass-through clicks

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    
    // Transparent window container
    if (ImGui::Begin("##ViewportMessages", nullptr, flags)) {
        
        // 1. Persistent HUD: Render Status (ALWAYS AT TOP)
        if (ctx.scene.initialized) {
            const bool in_rendered_mode = (viewport_settings.shading_mode == 2);

            std::string status_text;
            ImU32 text_col = IM_COL32(220, 220, 220, 255);

            if (in_rendered_mode) {
                // Path trace mode: show sample progress
                int current = ctx.render_settings.render_current_samples;
                int target = ctx.render_settings.render_target_samples;
                bool is_paused = ctx.render_settings.is_render_paused;

                std::string mode_tag = "[CPU]";
                if (ctx.backend_ptr) {
                    if (dynamic_cast<Backend::OptixBackend*>(ctx.backend_ptr) != nullptr) {
                        mode_tag = "[OptiX]";
                    } else if (dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr) != nullptr) {
                        mode_tag = "[Vulkan]";
                    }
                }

                if (current >= target && target > 0) {
                     status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples (Done)";
                     text_col = IM_COL32(100, 255, 100, 255);
                } else if (is_paused) {
                     status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples (Paused)";
                     text_col = IM_COL32(255, 180, 80, 255);
                } else {
                     status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples";
                }

                if (ctx.render_settings.avg_total_frame_time_ms > 0.0f) {
                    char perf_text[96];
                    snprintf(perf_text, sizeof(perf_text), " | %.0f ms/frame | %.1f fps",
                             ctx.render_settings.avg_total_frame_time_ms,
                             ctx.render_settings.avg_total_frame_fps);
                    status_text += perf_text;
                }
            } else {
                // Solid/Matcap/MaterialPreview: show viewport mode name
                switch (viewport_settings.shading_mode) {
                    case 0: status_text = "Solid Mode"; break;
                    case 1: status_text = "Material Preview"; break;
                    case 3: status_text = "Matcap Mode"; break;
                    default: status_text = "Viewport Mode"; break;
                }
                text_col = IM_COL32(180, 210, 255, 255);
            }

            const auto drawHudLine = [](const std::string& text, ImU32 color) {
                ImVec2 pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 200), text.c_str());
                ImGui::GetWindowDrawList()->AddText(pos, color, text.c_str());
                ImGui::Dummy(ImGui::CalcTextSize(text.c_str()));
            };

            drawHudLine(status_text, text_col);

            if (ctx.render_settings.show_scene_stats_hud) {
                drawHudLine("Scene tris: " + formatCompactCount(cached_scene_triangle_count), IM_COL32(235, 235, 235, 225));

                const size_t instance_count = InstanceManager::getInstance().getTotalInstanceCount();
                const size_t instance_triangle_count = InstanceManager::getInstance().getTotalTriangleCount();
                if (instance_count > 0) {
                    drawHudLine("Instances: " + formatCompactCount(instance_count), IM_COL32(235, 235, 235, 225));
                    drawHudLine("Instance tris: " + formatCompactCount(instance_triangle_count), IM_COL32(235, 235, 235, 225));
                }
            }

            ImGui::Dummy(ImVec2(0, 2)); // Small spacing
        }

        // 2. Persistent HUD: Selected Object (BELOW RENDER STATUS)
        if (ctx.selection.hasSelection()) {
            std::string sel_text = "Selected: " + ctx.selection.selected.name;
            ImVec2 pos = ImGui::GetCursorScreenPos();
            
            // Shadow
            ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x+1, pos.y+1), IM_COL32(0,0,0,200), sel_text.c_str());
            // Text (Orange)
            ImGui::GetWindowDrawList()->AddText(pos, IM_COL32(255, 180, 50, 255), sel_text.c_str());
            
            // Advance cursor to push messages down
            ImVec2 text_size = ImGui::CalcTextSize(sel_text.c_str());
            ImGui::Dummy(text_size); 
            ImGui::Dummy(ImVec2(0, 5)); // Extra spacing
        }

        // 3. Dynamic Messages (BELOW SELECTION)
        // Remove expired messages
        for (auto it = active_messages.begin(); it != active_messages.end();) {
            it->time_remaining -= dt;
            if (it->time_remaining <= 0.0f) {
                it = active_messages.erase(it);
            } else {
                ++it;
            }
        }
        
        // Draw messages
        for (const auto& msg : active_messages) {
            // Fade out
            float alpha = 1.0f;
            if (msg.time_remaining < 0.5f) {
                alpha = msg.time_remaining / 0.5f;
            }
            if (alpha < 0.0f) alpha = 0.0f;
            
            // Text Color with Alpha
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(msg.color.x, msg.color.y, msg.color.z, alpha));
            
            // Add subtle shadow for readability against 3D viewport
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0,0,0, (int)(200 * alpha)), msg.text.c_str());
            
            ImGui::TextUnformatted(msg.text.c_str());
            ImGui::PopStyleColor();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// ═════════════════════════════════════════════════════════════════════════════
// LENS INFO HUD (Top-left corner - shows current lens and settings)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawLensInfoHUD(UIContext& ctx) {
    if (!ctx.scene.camera) return;
    if (!viewport_settings.show_camera_hud) return;
    
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    
    Camera& cam = *ctx.scene.camera;
    
    // Position: Top-left corner, below menu bar
    float margin_left = 10.0f;
    float margin_top = 45.0f;  // Below menu bar
    float box_width = 200.0f;
    float box_height = 105.0f; // Increased for Backend badge
    
    ImVec2 box_pos(margin_left, margin_top);
    
    // Colors
    ImU32 col_bg = IM_COL32(20, 25, 30, 200);
    ImU32 col_border = IM_COL32(80, 100, 120, 200);
    ImU32 col_title = IM_COL32(100, 180, 255, 255);
    ImU32 col_label = IM_COL32(180, 180, 180, 255);
    ImU32 col_value = IM_COL32(255, 220, 100, 255);
    ImU32 col_fov = IM_COL32(100, 255, 150, 255);
    
    // Draw background
    draw_list->AddRectFilled(box_pos, ImVec2(box_pos.x + box_width, box_pos.y + box_height), col_bg, 5.0f);
    draw_list->AddRect(box_pos, ImVec2(box_pos.x + box_width, box_pos.y + box_height), col_border, 5.0f, 0, 1.5f);
    
    // Calculate focal length from FOV
    float fov = (float)cam.vfov;
    float focal_mm = 24.0f / (2.0f * tanf(fov * 0.5f * 3.14159f / 180.0f));
    
    // Determine lens type name
    const char* lens_type = "Custom";
    if (focal_mm < 20.0f) lens_type = "Ultra Wide";
    else if (focal_mm < 28.0f) lens_type = "Wide Angle";
    else if (focal_mm < 40.0f) lens_type = "Standard Wide";
    else if (focal_mm < 60.0f) lens_type = "Normal";
    else if (focal_mm < 100.0f) lens_type = "Portrait";
    else if (focal_mm < 200.0f) lens_type = "Telephoto";
    else if (focal_mm < 400.0f) lens_type = "Long Telephoto";
    else if (focal_mm < 800.0f) lens_type = "Super Telephoto";
    else lens_type = "Extreme Telephoto";
    
    // Get f-stop from preset
    float f_stop = 2.8f;
    if (cam.fstop_preset_index > 0 && cam.fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT) {
        f_stop = CameraPresets::FSTOP_PRESETS[cam.fstop_preset_index].f_number;
    }
    
    // Draw icon/title
    float x = box_pos.x + 8.0f;
    float y = box_pos.y + 5.0f;
    draw_list->AddText(ImVec2(x, y), col_title, "LENS");
    
    // Line 1: Focal Length + Type
    y += 18.0f;
    char line1[64];
    snprintf(line1, sizeof(line1), "%.0fmm %s", focal_mm, lens_type);
    draw_list->AddText(ImVec2(x, y), col_value, line1);
    
    // Line 2: FOV
    y += 16.0f;
    char line2[32];
    snprintf(line2, sizeof(line2), "FOV: %.1f", fov);
    draw_list->AddText(ImVec2(x, y), col_fov, line2);
    draw_list->AddText(ImVec2(x + 70, y), col_label, "\xC2\xB0");  // degree symbol
    
    // Line 3: Aperture and Focus
    y += 16.0f;
    char line3[64];
    snprintf(line3, sizeof(line3), "f/%.1f  Focus: %.2fm", f_stop, (float)cam.focus_dist);
    draw_list->AddText(ImVec2(x, y), col_label, line3);
    
    // Line 4: Badges (Mode and Backend)
    y += 18.0f;
    
    // Camera Mode badge
    const char* mode_label = "AUTO";
    ImU32 mode_color = IM_COL32(100, 200, 100, 255);
    
    if (cam.camera_mode == CameraMode::Pro) {
        mode_label = "PRO";
        mode_color = IM_COL32(100, 150, 255, 255);
    } else if (cam.camera_mode == CameraMode::Cinema) {
        mode_label = "CINEMA";
        mode_color = IM_COL32(255, 180, 80, 255);
    }
    
    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + 50, y + 14), mode_color, 3.0f);
    draw_list->AddText(ImVec2(x + 4, y), IM_COL32(0, 0, 0, 255), mode_label);
    
    // Backend Badge — reads the LIVE backend pointer (ctx.backend_ptr) rather than
    // the requested-engine flags. During async OptiX compile the flags flip the moment
    // the user picks OptiX in the combo, but the live backend is still Vulkan until
    // the worker thread finishes and the main loop swaps. Badge follows reality.
    const char* backend_label = "CPU";
    ImU32 backend_color = IM_COL32(150, 150, 150, 255);

    if (ctx.backend_ptr) {
        if (dynamic_cast<Backend::OptixBackend*>(ctx.backend_ptr) != nullptr) {
            backend_label = "OptiX";
            backend_color = IM_COL32(118, 185, 0, 255); // NVIDIA Green
        } else if (dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr) != nullptr) {
            backend_label = "Vulkan";
            backend_color = IM_COL32(230, 25, 35, 255); // Vulkan Red
        }
    }
    
    float bx = x + 55.0f;
    draw_list->AddRectFilled(ImVec2(bx, y), ImVec2(bx + 45, y + 14), backend_color, 3.0f);
    draw_list->AddText(ImVec2(bx + 4, y), IM_COL32(255, 255, 255, 255), backend_label);
    
    // Shake indicator (if active)
    if (cam.enable_camera_shake) {
        draw_list->AddRectFilled(ImVec2(x + 55, y), ImVec2(x + 105, y + 14), IM_COL32(200, 100, 50, 255), 3.0f);
        draw_list->AddText(ImVec2(x + 59, y), IM_COL32(255, 255, 255, 255), "SHAKE");
    }
}
