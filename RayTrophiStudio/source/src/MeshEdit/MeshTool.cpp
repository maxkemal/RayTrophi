/* MeshEdit/MeshTool.cpp */

#include "MeshEdit/MeshTool.h"

#include <algorithm>
#include <utility>

namespace MeshEdit {

void MeshOperationReport::addWarning(std::string code, std::string message) {
    diagnostics.push_back({std::move(code), std::move(message), true});
}

void MeshOperationReport::addError(std::string code, std::string message) {
    ok = false;
    diagnostics.push_back({std::move(code), std::move(message), false});
}

MeshToolRegistry& MeshToolRegistry::instance() {
    static MeshToolRegistry registry;
    return registry;
}

bool MeshToolRegistry::registerTool(MeshToolDescriptor descriptor, std::string* error) {
    if (descriptor.id.empty()) {
        if (error) *error = "mesh tool id is empty";
        return false;
    }
    if (find(descriptor.id) != nullptr) {
        if (error) *error = "mesh tool id already registered: " + descriptor.id;
        return false;
    }
    tools_.push_back(std::move(descriptor));
    return true;
}

bool MeshToolRegistry::unregisterTool(const std::string& id) {
    const auto it = std::find_if(tools_.begin(), tools_.end(),
        [&](const MeshToolDescriptor& tool) { return tool.id == id; });
    if (it == tools_.end()) return false;
    tools_.erase(it);
    return true;
}

const MeshToolDescriptor* MeshToolRegistry::find(const std::string& id) const {
    const auto it = std::find_if(tools_.begin(), tools_.end(),
        [&](const MeshToolDescriptor& tool) { return tool.id == id; });
    return it == tools_.end() ? nullptr : &*it;
}

std::vector<MeshToolDescriptor> MeshToolRegistry::list(
    MeshToolWorkspace workspace, bool include_unavailable) const {
    std::vector<MeshToolDescriptor> result;
    for (const auto& tool : tools_) {
        if (tool.workspace != workspace) continue;
        if (!include_unavailable && tool.availability != MeshToolAvailability::Implemented &&
            tool.availability != MeshToolAvailability::PreviewOnly) {
            continue;
        }
        result.push_back(tool);
    }
    return result;
}

void MeshToolRegistry::clear() {
    tools_.clear();
}

void registerBuiltInMeshTools() {
    auto& registry = MeshToolRegistry::instance();
    registry.clear();

    const MeshToolCapabilities editCpu{true, false, false, true, false, true};
    const MeshToolCapabilities editPreview{true, true, false, true, false, true};
    const MeshToolCapabilities profilePreview{true, true, false, true, false, false};

    registry.registerTool({
        "edit.transform", "Transform", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Vertex, MeshToolAvailability::Implemented,
        profilePreview, true, true, true,
        "Move selected editable elements on the canonical flat mesh."});
    registry.registerTool({
        "edit.extrude", "Extrude Faces", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Face, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Extrude selected polygon faces and publish the resulting flat mesh."});
    registry.registerTool({
        "edit.inset", "Inset Faces", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Face, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Inset selected polygon faces."});
    registry.registerTool({
        "edit.loop_cut", "Loop Cut", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Edge, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Insert a cut across a supported quad edge ring."});
    registry.registerTool({
        "edit.dissolve", "Dissolve Edges", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Edge, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Dissolve selected interior edges while preserving valid topology."});
    // ★ This was registered as Planned while SceneUI::bevelSelectedEdges had
    // been implemented for a long time. The catalogue understated the tool in
    // exactly the way it overstated the others: availability described intent,
    // not the code. A "planned" tool is hidden from the default listing, so
    // the one real rounding operator in the edit workspace was invisible to
    // every caller that asked what the application can do.
    registry.registerTool({
        "edit.edge_bevel", "Bevel Selected Edges", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Edge, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Bevel selected edges into a flat chamfer or a rounded profile."});
    registry.registerTool({
        "edit.dissolve_vertices", "Dissolve Vertices", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Vertex, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Dissolve selected vertices, merging the faces around them."});
    registry.registerTool({
        "edit.merge_vertices", "Merge Vertices To Centre", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Vertex, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Merge selected vertices into their common centre."});
    registry.registerTool({
        "edit.weld_vertices", "Weld Vertices By Distance", MeshToolWorkspace::Edit,
        MeshSelectionDomain::Vertex, MeshToolAvailability::Implemented,
        editCpu, true, true, true,
        "Weld selected vertices lying within an absolute distance of each other."});
    registry.registerTool({
        "profile.sweep", "Profile Sweep", MeshToolWorkspace::Profile,
        MeshSelectionDomain::Object, MeshToolAvailability::PreviewOnly,
        profilePreview, true, true, true,
        "Preview a mesh by sweeping an editable 2D profile along an editable curve path."});
    registry.registerTool({
        "profile.revolve", "Profile Revolve / Screw", MeshToolWorkspace::Profile,
        MeshSelectionDomain::Object, MeshToolAvailability::PreviewOnly,
        profilePreview, true, true, true,
        "Preview a rotational mesh from an editable closed radial profile."});
    registry.registerTool({
        "boolean.sdf", "SDF Boolean", MeshToolWorkspace::Boolean,
        MeshSelectionDomain::Object, MeshToolAvailability::Planned,
        {}, true, false, false,
        "Planned robust boolean using a general geometry SDF backend."});
}

} // namespace MeshEdit
