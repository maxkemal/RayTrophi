#include "MeshEditToolDock.h"

#include "MeshEdit/MeshTool.h"
#include "imgui.h"

namespace MeshEditUI {

void drawMeshEditContextHeader(const std::string& object_name,
                               const char* selection_mode,
                               size_t vertex_count,
                               size_t edge_count,
                               size_t face_count,
                               size_t selected_vertex_count,
                               size_t selected_edge_count,
                               size_t selected_face_count) {
    ImGui::SeparatorText("Edit Mesh");
    ImGui::Text("Object: %s", object_name.empty() ? "<none>" : object_name.c_str());
    ImGui::TextDisabled("Canonical: flat TriangleMesh / DNA SoA");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.36f, 0.86f, 0.68f, 1.0f), "CPU authoritative");

    ImGui::Text("Selection: %s", selection_mode ? selection_mode : "Object");
    ImGui::Text("V %zu / %zu    E %zu / %zu    F %zu / %zu",
                selected_vertex_count, vertex_count,
                selected_edge_count, edge_count,
                selected_face_count, face_count);

    size_t implemented = 0;
    size_t planned = 0;
    for (const auto& tool : MeshEdit::MeshToolRegistry::instance().list(
             MeshEdit::MeshToolWorkspace::Edit, true)) {
        if (tool.availability == MeshEdit::MeshToolAvailability::Implemented) ++implemented;
        if (tool.availability == MeshEdit::MeshToolAvailability::Planned) ++planned;
    }
    ImGui::TextDisabled("Tools: %zu ready  |  %zu planned  |  GPU preview available per tool", implemented, planned);
    ImGui::Separator();
}

} // namespace MeshEditUI
