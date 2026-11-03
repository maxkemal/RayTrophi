// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>

struct UIContext;
class PrincipledBSDF;

namespace MaterialUvSamplingUI {
bool draw(UIContext& ctx, PrincipledBSDF& material, uint16_t material_id,
          const std::string& object);
} // namespace MaterialUvSamplingUI
