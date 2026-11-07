// SPDX-License-Identifier: MIT
#pragma once

#include "UV/UvLayout.h"
#include "json.hpp"
#include <memory>

class Texture;

namespace UV {

struct PreviewTexture {
    std::string label;
    uint16_t material = 0;
    std::shared_ptr<Texture> texture;
};

// Main-thread read-only material channel inventory, also described by uv.info.
std::vector<PreviewTexture> previewTextures(const std::string& object, int mesh);

// Main thread only, including all read access. IPC must enqueue before calling.
bool inspect(const std::string& object, int mesh, Layout& layout, std::string& error);
uint64_t sceneRevision();
nlohmann::json request(const std::string& operation, const nlohmann::json& params);

} // namespace UV
