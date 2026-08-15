#include "Template/TemplateUiStateAdapter.h"

#include "Template/TemplateRegistry.h"
#include "scene_ui.h"

#include <unordered_map>

extern bool g_solid_viewport_active;

namespace raytrophi::templates {

TemplateUiApplyResult TemplateUiStateAdapter::apply(const TemplateUiState& state, SceneUI& ui) {
    TemplateUiApplyResult result;
    static const std::unordered_map<std::string, int> property_tabs{
        {"scene", 0}, {"render", 1}, {"terrain", 2}, {"water", 3},
        {"volumetric", 4}, {"simulation", 5}, {"world", 6},
        {"modeling", 7}, {"hair", 8}, {"system", 9}, {"paint", 10},
        {"scatter", 11}, {"stylize", 12}, {"sculpt", 13}};

    const auto property = property_tabs.find(state.properties_context);
    if (property != property_tabs.end()) {
        ui.active_properties_tab = property->second;
        ui.focus_properties_panel_next_frame = true;
    } else {
        result.warnings.push_back("properties context was not applied: " + state.properties_context);
    }

    ui.show_animation_panel = state.show_timeline;
    ui.show_terrain_graph = state.bottom_editor == "terrain";
    ui.show_geometry_graph = state.bottom_editor == "geometry";
    ui.show_material_graph = state.bottom_editor == "material";
    ui.show_anim_graph = state.bottom_editor == "anim_graph";
    if (state.bottom_editor == "dope_sheet" || state.bottom_editor == "graph_editor") {
        ui.show_animation_panel = true;
    } else if (state.bottom_editor != "none" && state.bottom_editor != "terrain" &&
               state.bottom_editor != "geometry" && state.bottom_editor != "material" &&
               state.bottom_editor != "anim_graph" && state.bottom_editor != "assets" &&
               state.bottom_editor != "console") {
        result.warnings.push_back("bottom editor was not applied: " + state.bottom_editor);
    }

    if (state.contextual_dock != "none") {
        result.warnings.push_back(
            "contextual dock activation is deferred until the target authoring mode is active: " +
            state.contextual_dock);
    }

    static const std::unordered_map<std::string, int> shading_modes{
        {"solid", 0}, {"material_preview", 1}, {"rendered", 2}, {"matcap", 3}};
    const auto shading = shading_modes.find(state.viewport_shading);
    if (shading != shading_modes.end()) {
        ui.viewport_settings.shading_mode = shading->second;
        g_solid_viewport_active = shading->second != 2;
    } else {
        result.warnings.push_back("viewport shading was not applied: " + state.viewport_shading);
    }
    result.applied = true;
    return result;
}

} // namespace raytrophi::templates
