// SPDX-License-Identifier: MIT
#pragma once

struct UIContext;
class TriangleMesh;

namespace UV {
// UV-only changes: refresh resident shading buffers without geometry-generation bumps.
void publishAttributeUpdate(UIContext& ctx, const TriangleMesh* mesh);
}
