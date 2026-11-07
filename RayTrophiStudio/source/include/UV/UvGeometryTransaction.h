// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>

struct UIContext;
class SceneHistory;
class TriangleMesh;
namespace DNA { class GeometryDetail; }

namespace UV {
// Main thread only. Retains the original geometry/delta state for exact pointer-swap undo.
void commitGeometry(UIContext& ctx, SceneHistory& history,
                    const std::shared_ptr<TriangleMesh>& mesh,
                    std::shared_ptr<DNA::GeometryDetail> after, const std::string& label);
} // namespace UV
