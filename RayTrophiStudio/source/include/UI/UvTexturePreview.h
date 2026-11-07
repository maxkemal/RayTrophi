// SPDX-License-Identifier: MIT
#pragma once

#include "imgui.h"
#include <memory>
#include <string>

class Texture;

namespace UvTexturePreview {
// UI thread only. Draws the raw UV tile at native texture resolution.
bool draw(const std::shared_ptr<Texture>& texture, ImVec2 top_left, ImVec2 bottom_right,
          float opacity, bool refresh, std::string& error);
void release();
}
