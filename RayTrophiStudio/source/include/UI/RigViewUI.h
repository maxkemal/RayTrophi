#pragma once
#include "scene_data.h"
struct UIContext;
namespace RigUI {
void drawBoneTree(UIContext&, const SceneData::ImportedModelContext&, int);
void drawModelControls(UIContext&, const SceneData::ImportedModelContext&);
}
