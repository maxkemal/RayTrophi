#include "MeshEditToolDock.h"

#include "MeshEdit/MeshTool.h"
#include "imgui.h"

namespace MeshEditUI {

void drawMeshEditContextHeader(const std::string& /*object_name*/,
                               const char* /*selection_mode*/,
                               size_t /*vertex_count*/,
                               size_t /*edge_count*/,
                               size_t /*face_count*/,
                               size_t /*selected_vertex_count*/,
                               size_t /*selected_edge_count*/,
                               size_t /*selected_face_count*/) {
    // Header moved to Viewport HUD overlay to maximize vertical space for tools in right dock.
}

} // namespace MeshEditUI
