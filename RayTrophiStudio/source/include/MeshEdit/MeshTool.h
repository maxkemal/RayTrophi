/*
 * MeshEdit/MeshTool.h
 *
 * Shared, UI-independent mesh tool metadata and operation diagnostics.
 * Geometry operations remain responsible for mutating the canonical flat
 * TriangleMesh/DNA SoA path. This registry only describes capabilities and
 * gives UI, scripting, IPC and addons one discovery contract.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace MeshEdit {

enum class MeshToolWorkspace : uint8_t {
    Edit,
    Profile,
    Curve,
    Surface,
    Boolean,
    Cleanup,
};

enum class MeshSelectionDomain : uint8_t {
    Object,
    Vertex,
    Edge,
    Face,
    Polygon,
    CurvePoint,
    SurfaceControl,
};

enum class MeshToolAvailability : uint8_t {
    Implemented,
    PreviewOnly,
    Planned,
    Disabled,
};

struct MeshToolCapabilities {
    bool cpu = false;
    bool gpu_preview = false;
    bool gpu_commit = false;
    bool deterministic = true;
    bool supports_cancel = false;
    bool undoable = true;
};

struct MeshToolDescriptor {
    std::string id;              // Stable IPC/addon id, e.g. "edit.extrude"
    std::string display_name;
    MeshToolWorkspace workspace = MeshToolWorkspace::Edit;
    MeshSelectionDomain selection_domain = MeshSelectionDomain::Object;
    MeshToolAvailability availability = MeshToolAvailability::Planned;
    MeshToolCapabilities capabilities;
    bool previewable = false;
    bool scriptable = false;
    bool ipc_exposed = false;
    std::string summary;
};

struct MeshOperationCounts {
    uint64_t vertices_changed = 0;
    uint64_t edges_changed = 0;
    uint64_t faces_changed = 0;
    uint64_t polygons_changed = 0;
    uint64_t triangles_changed = 0;
};

struct MeshOperationDiagnostic {
    std::string code;
    std::string message;
    bool warning = false;
};

struct MeshOperationReport {
    bool ok = false;
    std::string operation_id;
    uint64_t revision = 0;
    std::string undo_group;
    MeshOperationCounts changed;
    std::vector<MeshOperationDiagnostic> diagnostics;

    void addWarning(std::string code, std::string message);
    void addError(std::string code, std::string message);
};

class MeshToolRegistry {
public:
    static MeshToolRegistry& instance();

    // Registration is intended for application startup and addon loading.
    // Duplicate ids are rejected to keep IPC/tool discovery deterministic.
    bool registerTool(MeshToolDescriptor descriptor, std::string* error = nullptr);
    bool unregisterTool(const std::string& id);
    const MeshToolDescriptor* find(const std::string& id) const;
    std::vector<MeshToolDescriptor> list(
        MeshToolWorkspace workspace,
        bool include_unavailable = false) const;
    void clear();

private:
    std::vector<MeshToolDescriptor> tools_;
};

// Registers only operations that are currently implemented and safe to expose.
// Planned tools must not be advertised to agents or IPC callers as executable.
void registerBuiltInMeshTools();

} // namespace MeshEdit
