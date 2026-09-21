#pragma once

#include "RtIpcTemplates.h"

#include <functional>
#include <string>

// ★ Mesh editing mutates scene topology and records undo, so it must NOT run
// on the IPC thread. The adapter never calls rtapi directly; every method
// below is wrapped in the caller's enqueue so the frame loop owns the
// mutation, the same contract scene.* mutations already follow.

// Handles mesh.edit.* (selection, state) and mesh.* polygon operators.
// Returns true when the method belongs to this adapter, including validation
// errors encoded in out_result.
bool dispatchMeshEditIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue_query,
                         nlohmann::json& out_result);
