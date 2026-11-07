// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <string>

struct UIContext;

namespace UvEditorUI {
void open(const std::string& object = {});
bool capturesShortcuts();
// The SceneUI host owns docking. The UV view never reaches into host internals.
void draw(UIContext& ctx, const std::function<void(const char*)>& dock_to_bottom);
} // namespace UvEditorUI
