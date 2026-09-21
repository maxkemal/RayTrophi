#pragma once
#include "Animation/RigView.h"
struct UIContext;
namespace RigUI {
void drawRigRestGizmo(UIContext&, int shadingMode, bool enabled, bool& hit);
void applyRigDragPreview(const SceneData&, std::vector<RigAuthoring::BoneView>&);
}
