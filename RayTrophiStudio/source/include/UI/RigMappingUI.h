#pragma once
#include <functional>
struct UIContext;
namespace RigUI {
// Reuses the existing dock/float Animation editor surface.
void drawAnimationWorkspace(UIContext&, const std::function<void()>& drawGraph);
}
