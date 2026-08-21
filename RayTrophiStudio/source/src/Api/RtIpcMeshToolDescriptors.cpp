#include "RtIpcMethodRegistry.h"

static const MethodParam params_mesh_tools_list[] = {
    {"workspace", "string", false, "edit, profile, curve, surface, boolean or cleanup", "edit", "edit|profile|curve|surface|boolean|cleanup"},
    {"include_unavailable", "bool", false, "Include planned and disabled tools", "false", nullptr},
};
static const MethodDescriptor desc_mesh_tools_list = {
    "mesh.tools.list", "mesh",
    "List mesh modeling tools available in a workspace",
    "Only implemented or preview-only tools are returned by default. Planned tools are never executable.",
    "read", "any", false, "MeshToolDescriptor[]",
    "mesh|tools|edit|profile|curve|surface|boolean|cleanup|discovery",
    "mesh.tools.describe", nullptr, "mesh.tools.describe", "mesh.tools.describe", nullptr,
    params_mesh_tools_list, 2, true
};
static const MethodRegistration reg_mesh_tools_list(desc_mesh_tools_list);

static const MethodParam params_mesh_tools_describe[] = {
    {"tool", "string", true, "Stable mesh tool id, e.g. edit.extrude", nullptr, nullptr},
};
static const MethodDescriptor desc_mesh_tools_describe = {
    "mesh.tools.describe", "mesh",
    "Describe one mesh modeling tool and its CPU/GPU capabilities",
    "Use this before preview or commit. A planned tool is metadata only and cannot be executed.",
    "read", "any", false, "MeshToolDescriptor",
    "mesh|tools|describe|capabilities|agent",
    "mesh.tools.list", nullptr, nullptr, nullptr, nullptr,
    params_mesh_tools_describe, 1, true
};
static const MethodRegistration reg_mesh_tools_describe(desc_mesh_tools_describe);

static const MethodParam params_mesh_asset_validate[] = {
    {"object", "string", true, "Flat TriangleMesh object name", nullptr, nullptr},
};
static const MethodDescriptor desc_mesh_asset_validate = {
    "mesh.asset.validate", "mesh",
    "Validate a canonical flat SoA mesh without mutating the scene",
    "Checks finite positions/normals, index bounds, triangle degeneracy and index alignment.",
    "read", "any", false, "MeshValidationInfo",
    "mesh|asset|validate|diagnostics|flat|soa",
    nullptr, nullptr, nullptr, nullptr, nullptr,
    params_mesh_asset_validate, 1, true
};
static const MethodRegistration reg_mesh_asset_validate(desc_mesh_asset_validate);

static const MethodParam params_mesh_operation_plan[] = {
    {"object", "string", true, "Flat TriangleMesh object name", nullptr, nullptr},
    {"tool", "string", true, "Stable mesh tool id", nullptr, nullptr},
    {"backend", "string", false, "auto, cpu or gpu", "auto", "auto|cpu|gpu"},
    {"preview", "bool", false, "Prepare a non-mutating preview", "false", nullptr},
    {"commit", "bool", false, "Prepare a commit; mutation is not performed by planning", "false", nullptr},
};
static const MethodDescriptor desc_mesh_operation_plan = {
    "mesh.operation.plan", "mesh",
    "Preflight a mesh operation against the canonical flat SoA mesh",
    "Side-effect free. Validates tool availability, phase, backend and object state before execution.",
    "read", "any", false, "MeshOperationPlanInfo",
    "mesh|operation|plan|preflight|agent|cpu|gpu",
    "mesh.tools.describe", nullptr, "mesh.tools.describe", "mesh.asset.validate", nullptr,
    params_mesh_operation_plan, 5, true
};
static const MethodRegistration reg_mesh_operation_plan(desc_mesh_operation_plan);

static const MethodDescriptor desc_mesh_operation_self_test = {
    "mesh.operation.self_test", "mesh",
    "Run the deterministic half-edge transaction self-test",
    "Read-only core test; it builds an in-memory quad, extrudes it transactionally and validates the result.",
    "read", "any", false, "MeshOperationSelfTest",
    "mesh|operation|self_test|topology|diagnostics",
    "mesh.operation.plan", nullptr, nullptr, nullptr, nullptr,
    nullptr, 0, true
};
static const MethodRegistration reg_mesh_operation_self_test(desc_mesh_operation_self_test);

static const MethodParam params_mesh_operation_commit_positions[] = {
    {"object", "string", true, "Flat TriangleMesh object name", nullptr, nullptr},
    {"positions", "float[vertex_count][3]", true, "Local/bind-space vertex positions", nullptr, nullptr},
};
static const MethodDescriptor desc_mesh_operation_commit_positions = {
    "mesh.operation.commit_positions", "mesh",
    "Commit flat mesh positions through an undoable operation",
    "Writes P_orig, rebakes P/N, validates the flat SoA mesh, schedules scene refresh and records SceneHistory.",
    "write", "Scene|Geometry", true, "MeshOperationResult",
    "mesh|operation|commit|positions|undo|flat|soa",
    "mesh.operation.plan", nullptr, "mesh.operation.plan", "mesh.asset.validate", nullptr,
    params_mesh_operation_commit_positions, 2, true
};
static const MethodRegistration reg_mesh_operation_commit_positions(desc_mesh_operation_commit_positions);
